#!/usr/bin/env python3
"""
Готовит данные карты для приложения viision из выгрузки OpenStreetMap (.osm.pbf).

Запуск:  python3 build_country.py filtered.osm.pbf out IT
  filtered.osm.pbf — страна с Geofabrik, где оставлены только дороги и границы
                     (osmium tags-filter italy.osm.pbf w/highway r/boundary=administrative)
  out              — папка для результата
  IT               — код страны (тег ISO3166-1 у границы страны)

Что получается в папке out:
  <код>-index.json       — маленький общий файл: все области страны (страна, регионы, провинции,
                           коммуны, кварталы), общая длина улиц в каждой, упрощённые контуры
                           страны/регионов/провинций, имена файлов провинций.
  <код>-comuni.vpkg      — упрощённые (~50 м) границы всех коммун: по ним приложение видит,
                           в каких коммунах ты был. Качается вместе с общим файлом.
  province-<номер>.vpkg  — файл одной провинции: точные границы её коммун и кварталов
                           и улицы каждой коммуны. Приложение качает его, когда добавляешь город.
                           Где провинций нет (Валле-д'Аоста), такой файл делается на весь регион.

Формат файлов .vpkg (числа little-endian):
  "VPKG", версия UInt32 (=1), число записей UInt32
  записи: вид UInt8 (0 — границы коммун и кварталов, 1 — улицы коммуны), номер Int64,
          начало UInt64, размер в файле UInt32, размер после распаковки UInt32
  дальше сами записи, каждая сжата (raw DEFLATE, как NSData .zlib в iOS)

Улицы коммуны — формат VSTR версии 2 (как streets-<id>.bin в приложении), только точки
записаны разницей от предыдущей точки той же линии: так файл сжимается в 2–3 раза лучше.

Границы — формат VARE версии 1:
  "VARE", версия UInt32, число областей UInt32
  каждая область: номер Int64, номер родителя Int64 (0 — нет), уровень UInt8,
                  название (UInt16 длина + UTF-8), число колец UInt32,
                  каждое кольцо: число точек UInt32, точки Int32 широта/долгота × 1 000 000
                  (первая точка как есть, остальные — разницей от предыдущей)
"""
import json
import os
import struct
import sys
import time
import zlib
from array import array

import numpy as np
import osmium
import shapely
from shapely import STRtree

# Порядок важен: тот же список — StreetWay.kinds в Swift.
KINDS = [
    "residential", "living_street", "pedestrian", "unclassified", "road",
    "tertiary", "tertiary_link", "secondary", "secondary_link", "primary", "primary_link",
    "service", "footway", "cycleway", "path", "track",
]
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
BAD_SERVICE = {"driveway", "parking_aisle", "drive-through", "fuel", "emergency_access"}
BAD_FOOTWAY = {"sidewalk", "crossing", "traffic_island"}
BAD_ACCESS = {"private", "no", "customers", "emergency", "delivery", "permit"}
LEVELS = {2, 4, 6, 8, 10}          # страна, регион, провинция, коммуна, квартал
OUTLINE_TOLERANCE = 0.002          # градусы (~200 м): упрощение контуров в общем файле
COMUNI_TOLERANCE = 0.0005          # градусы (~50 м): упрощение границ коммун в файле коммун
KY = 6_371_000.0 * np.pi / 180     # метров в градусе широты — как в приложении


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def keep(tags):
    """Какие линии считаем улицами — те же правила, что в приложении."""
    highway = tags.get("highway")
    if highway not in KIND_INDEX:
        return False
    if tags.get("area") == "yes" or tags.get("indoor") == "yes":
        return False
    if tags.get("access") in BAD_ACCESS:
        return False
    if highway == "service" and tags.get("service") in BAD_SERVICE:
        return False
    if highway == "footway" and tags.get("footway") in BAD_FOOTWAY:
        return False
    return True


# ---------- 1. Границы ----------

def read_areas(path):
    """Все административные границы нужных уровней, собранные osmium в многоугольники."""
    areas = []
    processor = (osmium.FileProcessor(path)
                 .with_areas(osmium.filter.TagFilter(("boundary", "administrative")))
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA)))
    for area in processor:
        if area.from_way():
            continue
        level = area.tags.get("admin_level", "")
        name = area.tags.get("name")
        if not name or not level.isdigit() or int(level) not in LEVELS:
            continue
        polygons = []
        for outer in area.outer_rings():
            shell = [(n.lon, n.lat) for n in outer]
            holes = [[(n.lon, n.lat) for n in inner] for inner in area.inner_rings(outer)]
            if len(shell) >= 4:
                polygons.append(shapely.Polygon(shell, holes))
        if not polygons:
            continue
        geom = polygons[0] if len(polygons) == 1 else shapely.MultiPolygon(polygons)
        if not geom.is_valid:
            geom = geom.buffer(0)
        areas.append({
            "id": area.orig_id(), "name": name, "level": int(level), "parentID": None,
            "iso": area.tags.get("ISO3166-1") or area.tags.get("ISO3166-1:alpha2"),
            "iso2": area.tags.get("ISO3166-2", ""),
            "geom": geom,
        })
    return areas


def attach(children, parents):
    """Родитель области — та, в которой лежит точка внутри неё. Возвращает тех, кому нашёлся родитель."""
    if not children or not parents:
        return []
    tree = STRtree([p["geom"] for p in parents])
    points = [shapely.point_on_surface(c["geom"]) for c in children]
    child_index, parent_index = tree.query(points, predicate="within")
    found = []
    seen = set()
    for c, p in zip(child_index, parent_index):
        if c in seen:
            continue
        seen.add(c)
        children[c]["parentID"] = parents[p]["id"]
        found.append(children[c])
    return found


def build_hierarchy(areas, code):
    # Регионы страны узнаём по тегу ISO3166-2 («IT-45» — Эмилия-Романья). Границу самой страны
    # не берём: у неё есть морская часть далеко от берега, и в выгрузке она часто неполная.
    regions = [a for a in areas if a["level"] == 4 and a["iso2"].startswith(code + "-")]
    if not regions:
        sys.exit(f"No regions with ISO3166-2={code}-… found")
    official = next((a for a in areas if a["level"] == 2 and a["iso"] == code), None)
    country = {"id": official["id"] if official else 0, "name": official["name"] if official else code,
               "level": 2, "parentID": None, "geom": shapely.union_all([r["geom"] for r in regions])}
    for r in regions:
        r["parentID"] = country["id"]
    provinces = attach([a for a in areas if a["level"] == 6], regions)
    comuni_all = [a for a in areas if a["level"] == 8]
    comuni = attach(comuni_all, provinces)
    # Коммуны вне провинций (например, Валле-д'Аоста, где провинций нет) — сразу к региону.
    orphans = [c for c in comuni_all if c["parentID"] is None]
    comuni += attach(orphans, regions)
    districts = attach([a for a in areas if a["level"] == 10], comuni)
    return country, regions, provinces, comuni, districts


def areas_file(areas, tolerance=0.0):
    """Границы областей в формате VARE: точные или упрощённые (tolerance в градусах)."""
    out = bytearray(b"VARE") + struct.pack("<II", 1, len(areas))
    for a in areas:
        raw = a["name"].encode("utf-8")
        rings = []
        geom = a["geom"]
        if tolerance > 0:
            geom = shapely.simplify(geom, tolerance, preserve_topology=True)
        for polygon in shapely.get_parts(geom):
            if polygon.is_empty:
                continue
            for ring in [polygon.exterior, *polygon.interiors]:
                rings.append(np.asarray(ring.coords))
        out += struct.pack("<qqB", a["id"], a["parentID"] or 0, a["level"])
        out += struct.pack("<H", len(raw)) + raw + struct.pack("<I", len(rings))
        for coords in rings:
            lat = np.round(coords[:, 1] * 1_000_000).astype(np.int64)
            lon = np.round(coords[:, 0] * 1_000_000).astype(np.int64)
            points = np.empty(2 * len(coords), dtype="<i4")
            points[0::2] = np.diff(lat, prepend=0)
            points[1::2] = np.diff(lon, prepend=0)
            out += struct.pack("<I", len(coords)) + points.tobytes()
    return bytes(out)


def outline(geom):
    """Упрощённый контур для общего файла: [[ [широта, долгота], … ], …]."""
    simple = shapely.simplify(geom, OUTLINE_TOLERANCE, preserve_topology=True)
    rings = []
    for polygon in shapely.get_parts(simple):
        if polygon.is_empty:
            continue
        for ring in [polygon.exterior, *polygon.interiors]:
            rings.append([[round(y, 5), round(x, 5)] for x, y in ring.coords])
    return rings


# ---------- 2. Улицы ----------

class Streets:
    """Все подходящие линии страны в плоских массивах — так они занимают мало памяти."""

    def __init__(self):
        self.ids = array("q")
        self.name_ref = array("i")      # номер названия в self.names или −1
        self.kinds = array("b")
        self.counts = array("I")
        self.lats = array("d")
        self.lons = array("d")
        self.names = []
        self.name_index = {}

    def read(self, path):
        processor = (osmium.FileProcessor(path).with_locations()
                     .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                     .with_filter(osmium.filter.KeyFilter("highway")))
        for way in processor:
            tags = way.tags
            if not keep(tags):
                continue
            lat, lon = [], []
            for node in way.nodes:
                location = node.location
                if not location.valid():
                    break
                lat.append(location.lat)
                lon.append(location.lon)
            else:
                if len(lat) < 2:
                    continue
                name = tags.get("name")
                if name is None:
                    self.name_ref.append(-1)
                else:
                    if name not in self.name_index:
                        self.name_index[name] = len(self.names)
                        self.names.append(name)
                    self.name_ref.append(self.name_index[name])
                self.ids.append(way.id)
                self.kinds.append(KIND_INDEX[tags.get("highway")])
                self.counts.append(len(lat))
                self.lats.extend(lat)
                self.lons.extend(lon)
        self.count = len(self.ids)
        self.counts_np = np.frombuffer(self.counts, dtype=np.uint32).astype(np.int64)
        self.offsets = np.zeros(self.count + 1, dtype=np.int64)
        np.cumsum(self.counts_np, out=self.offsets[1:])
        self.lat_np = np.frombuffer(self.lats, dtype=np.float64)
        self.lon_np = np.frombuffer(self.lons, dtype=np.float64)
        # Микроградусы — как в файле приложения.
        self.lat_micro = np.round(self.lat_np * 1_000_000).astype(np.int64)
        self.lon_micro = np.round(self.lon_np * 1_000_000).astype(np.int64)

    def lines(self, start, stop):
        """Линии shapely (x — долгота, y — широта) для линий с номерами start..stop−1."""
        a, b = self.offsets[start], self.offsets[stop]
        coords = np.column_stack([self.lon_np[a:b], self.lat_np[a:b]])
        index = np.repeat(np.arange(stop - start), self.counts_np[start:stop])
        return shapely.linestrings(coords, indices=index)

    def lines_for(self, indices):
        """Линии shapely для произвольного списка номеров линий — одним вызовом, без цикла."""
        indices = np.asarray(indices, dtype=np.int64)
        counts = self.counts_np[indices]
        starts = self.offsets[indices]
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        positions = np.repeat(starts, counts) + within
        coords = np.column_stack([self.lon_np[positions], self.lat_np[positions]])
        return shapely.linestrings(coords, indices=np.repeat(np.arange(len(indices)), counts))


def assign(streets, comuni, chunk=400_000):
    """Какие линии задевают какую коммуну. Линия на границе попадает в обе."""
    tree = STRtree([c["geom"] for c in comuni])
    per_comune = [[] for _ in comuni]
    for start in range(0, streets.count, chunk):
        stop = min(streets.count, start + chunk)
        lines = streets.lines(start, stop)
        line_index, comune_index = tree.query(lines, predicate="intersects")
        for l, c in zip(line_index.tolist(), comune_index.tolist()):
            per_comune[c].append(start + l)
        log(f"assigned {stop} / {streets.count} ways")
    return per_comune


def to_meters(geom, lat0, lon0):
    kx = KY * np.cos(np.radians(lat0))
    return shapely.transform(geom, lambda xy: np.column_stack(((xy[:, 0] - lon0) * kx, (xy[:, 1] - lat0) * KY)))


def measure(streets, way_indices, polygon):
    """Длина улиц внутри многоугольника (м) и сколько улиц с названием туда попадает."""
    if not way_indices:
        return 0.0, 0
    lat0 = (polygon.bounds[1] + polygon.bounds[3]) / 2
    lon0 = (polygon.bounds[0] + polygon.bounds[2]) / 2
    area = to_meters(polygon, lat0, lon0)
    shapely.prepare(area)
    lines = to_meters(streets.lines_for(way_indices), lat0, lon0)
    lengths = shapely.length(lines)
    inside = shapely.contains_properly(area, lines)
    crossing = ~inside
    if crossing.any():
        lengths[crossing] = shapely.length(shapely.intersection(lines[crossing], area))
    total = float(lengths.sum())
    named = {streets.name_ref[i] for i, length in zip(way_indices, lengths) if length > 0 and streets.name_ref[i] >= 0}
    return total, len(named)


def streets_file(streets, way_indices):
    """Улицы коммуны в формате VSTR версии 2 (точки — разницей от предыдущей)."""
    local_names, local_index = [], {}
    for i in way_indices:
        ref = streets.name_ref[i]
        if ref >= 0 and ref not in local_index:
            local_index[ref] = len(local_names)
            local_names.append(streets.names[ref])
    out = bytearray(b"VSTR")
    out += struct.pack("<II", 2, len(local_names))
    for name in local_names:
        raw = name.encode("utf-8")
        out += struct.pack("<H", len(raw)) + raw
    out += struct.pack("<I", len(way_indices))
    for i in way_indices:
        a, b = streets.offsets[i], streets.offsets[i + 1]
        lat, lon = streets.lat_micro[a:b], streets.lon_micro[a:b]
        dlat = np.diff(lat, prepend=0)
        dlon = np.diff(lon, prepend=0)
        points = np.empty(2 * (b - a), dtype="<i4")
        points[0::2] = dlat
        points[1::2] = dlon
        ref = streets.name_ref[i]
        out += struct.pack("<qiBI", streets.ids[i], local_index[ref] if ref >= 0 else -1,
                           streets.kinds[i], b - a)
        out += points.tobytes()
    return bytes(out)


def deflate(data):
    packer = zlib.compressobj(9, zlib.DEFLATED, -15)     # raw DEFLATE, без заголовков
    return packer.compress(data) + packer.flush()


def write_package(path, entries):
    """entries: список (вид, номер, данные). Каждая запись сжимается отдельно."""
    blobs = [(kind, ident, deflate(data), len(data)) for kind, ident, data in entries]
    header_size = 12 + len(blobs) * (1 + 8 + 8 + 4 + 4)
    out = bytearray(b"VPKG") + struct.pack("<II", 1, len(blobs))
    offset = header_size
    for kind, ident, blob, raw in blobs:
        out += struct.pack("<BqQII", kind, ident, offset, len(blob), raw)
        offset += len(blob)
    for _, _, blob, _ in blobs:
        out += blob
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


# ---------- Всё вместе ----------

def main():
    source, out_dir, code = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    log("reading boundaries")
    areas = read_areas(source)
    country, regions, provinces, comuni, districts = build_hierarchy(areas, code)
    log(f"{country['name']}: {len(regions)} regions, {len(provinces)} provinces, "
        f"{len(comuni)} municipalities, {len(districts)} districts")

    log("reading streets")
    streets = Streets()
    streets.read(source)
    log(f"{streets.count} ways, {len(streets.lats)} points")

    per_comune = assign(streets, comuni)

    # Длины улиц: коммуны, их кварталы; выше — суммы.
    log("measuring")
    totals = {}
    districts_of = {}
    for d in districts:
        districts_of.setdefault(d["parentID"], []).append(d)
    for n, (comune, ways) in enumerate(zip(comuni, per_comune)):
        ways.sort()
        totals[comune["id"]] = measure(streets, ways, comune["geom"])
        for d in districts_of.get(comune["id"], []):
            totals[d["id"]] = measure(streets, ways, d["geom"])
        if n % 500 == 0:
            log(f"measured {n} / {len(comuni)}")
    by_id = {a["id"]: a for a in [country, *regions, *provinces, *comuni, *districts]}
    for level in (6, 4, 2):
        for a in [x for x in by_id.values() if x["level"] == level]:
            km, named = 0.0, 0
            for child in by_id.values():
                if child["parentID"] == a["id"] and child["id"] in totals:
                    km += totals[child["id"]][0]
                    named += totals[child["id"]][1]
            totals[a["id"]] = (km, named)

    # Регион, к которому относится коммуна (через провинцию или напрямую).
    def region_of(area):
        while area and area["level"] != 4:
            area = by_id.get(area["parentID"])
        return area

    # Файл провинции: коммуны и кварталы, чей «хозяин» — эта провинция
    # (или регион, если провинций нет, как в Валле-д'Аосте).
    def owner_of(area):
        while area and area["level"] not in (6, 4):
            area = by_id.get(area["parentID"])
        return area

    log("writing province packages")
    packages = {}
    owners = {}
    for c in comuni:
        owners.setdefault(owner_of(c)["id"], []).append(c)
    index_of = {id(c): k for k, c in enumerate(comuni)}
    for owner_id, owner_comuni in owners.items():
        owner_comuni_ids = {c["id"] for c in owner_comuni}
        owner_districts = [d for d in districts if d["parentID"] in owner_comuni_ids]
        entries = [(0, owner_id, areas_file(owner_comuni + owner_districts))]
        for c in owner_comuni:
            entries.append((1, c["id"], streets_file(streets, per_comune[index_of[id(c)]])))
        name = f"province-{owner_id}.vpkg"
        packages[owner_id] = (name, write_package(os.path.join(out_dir, name), entries))
        log(f"{by_id[owner_id]['name']}: {len(owner_comuni)} municipalities, {packages[owner_id][1] / 1e6:.1f} MB")

    log("writing municipality outlines")
    comuni_name = f"{code.lower()}-comuni.vpkg"
    comuni_size = write_package(os.path.join(out_dir, comuni_name),
                                [(0, country["id"], areas_file(comuni, COMUNI_TOLERANCE))])
    log(f"{comuni_name}: {comuni_size / 1e6:.1f} MB")

    log("writing index")
    index_areas = []
    for a in [country, *regions, *provinces, *comuni, *districts]:
        km, named = totals.get(a["id"], (0.0, 0))
        item = {"id": a["id"], "name": a["name"], "level": a["level"], "parentID": a["parentID"],
                "total": round(km, 1), "streets": named}
        if a["level"] in (2, 4, 6):
            item["outline"] = outline(a["geom"])
        if a["id"] in packages:
            item["package"], item["packageSize"] = packages[a["id"]]
        index_areas.append(item)
    index = {"version": 2, "country": code, "name": country["name"],
             "date": time.strftime("%Y-%m-%d", time.gmtime()),
             "comuniFile": comuni_name, "comuniFileSize": comuni_size, "areas": index_areas}
    index_path = os.path.join(out_dir, f"{code.lower()}-index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, separators=(",", ":"), ensure_ascii=False)
    total_mb = sum(size for _, size in packages.values()) / 1e6
    log(f"index {os.path.getsize(index_path) / 1e6:.1f} MB, packages {total_mb:.0f} MB, "
        f"streets {totals[country['id']][0] / 1000:.0f} km")


if __name__ == "__main__":
    main()
