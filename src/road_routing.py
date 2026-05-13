"""OSM 도로망 기반 라우팅

트램 정거장 시퀀스를 실제 도로를 따라 연결하는 LineString 좌표를 생성.
대전 트램은 노면전차라 대부분 도로 위를 다니므로 도로망 shortest path 근사가 적절.

캐싱: 첫 호출 시 OSM에서 대전 도로망 다운로드 후
      data/processed/daejeon_road.graphml에 저장. 이후엔 캐시 사용.
"""
from pathlib import Path
import osmnx as ox
import networkx as nx
from src.config import DATA_PROCESSED

# 대전 도로망 bbox (lon_min, lat_min, lon_max, lat_max)
DAEJEON_BBOX = (127.28, 36.18, 127.55, 36.50)
GRAPH_CACHE = DATA_PROCESSED / "daejeon_road.graphml"
ARTERIAL_CACHE = DATA_PROCESSED / "daejeon_arterial.graphml"

# 트램이 다닐 만한 주간선도로 (한국 도시 트램 노면전차 기준)
ARTERIAL_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'


def get_road_graph():
    """대전 전체 도로망 (모든 차량도로 포함)"""
    if GRAPH_CACHE.exists():
        return ox.load_graphml(GRAPH_CACHE)
    print(f"📡 OSM 전체 도로망 다운로드 중...")
    G = ox.graph_from_bbox(bbox=DAEJEON_BBOX, network_type="drive", simplify=True)
    ox.save_graphml(G, GRAPH_CACHE)
    print(f"✅ 캐시 저장: {GRAPH_CACHE}")
    return G


def get_arterial_graph():
    """대전 주간선도로만 (trunk/primary/secondary/tertiary).
    트램은 골목길을 다니지 않으므로 shortest path를 주간선도로 위에서만 계산.
    """
    if ARTERIAL_CACHE.exists():
        return ox.load_graphml(ARTERIAL_CACHE)
    print(f"📡 OSM 주간선도로 다운로드 중...")
    G = ox.graph_from_bbox(
        bbox=DAEJEON_BBOX,
        custom_filter=ARTERIAL_FILTER,
        simplify=True,
    )
    ox.save_graphml(G, ARTERIAL_CACHE)
    print(f"✅ 캐시 저장: {ARTERIAL_CACHE} ({G.number_of_nodes()} 노드)")
    return G


def snap_points_to_road(latlon_seq, G=None, max_distance_m=300):
    """각 점을 가장 가까운 도로 노드 좌표로 스냅 (간선 path 안 만듦).

    Args:
        latlon_seq: [(lat, lon), ...]
        G: 도로망 그래프 (없으면 주간선도로 자동 로드)
        max_distance_m: 이 거리 이상 떨어진 점은 원본 유지 (도로 없는 외곽 보호)

    Returns:
        [(lat, lon), ...] — 도로 위로 스냅된 점들 (또는 원본)
    """
    if G is None:
        G = get_arterial_graph()
    from shapely.geometry import Point
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    out = []
    for lat, lon in latlon_seq:
        n = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        node_lat, node_lon = G.nodes[n]["y"], G.nodes[n]["x"]
        _, _, dist = geod.inv(lon, lat, node_lon, node_lat)
        if dist <= max_distance_m:
            out.append((node_lat, node_lon))
        else:
            out.append((lat, lon))  # 너무 멀면 원본 유지
    return out


def snap_path_to_road(latlon_sequence, G=None, max_detour_ratio=1.8):
    """좌표 시퀀스를 도로망 shortest path로 연결 + backtrack/우회 가드.

    각 정거장 사이의 shortest path 길이가 직선거리의 max_detour_ratio 배 초과면
    그 구간은 직선으로 fallback (트램 노선은 우회 안 한다는 가정).

    Args:
        latlon_sequence: [(lat, lon), ...]
        G: 도로망 그래프
        max_detour_ratio: 도로 path / 직선 거리 비율 한도

    Returns:
        [(lat, lon), ...] 도로 따라가는 조밀 좌표 (우회 구간은 직선)
    """
    if G is None:
        G = get_arterial_graph()
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    nodes = [ox.distance.nearest_nodes(G, X=lon, Y=lat) for lat, lon in latlon_sequence]
    full_coords = [latlon_sequence[0]]

    for i in range(len(nodes) - 1):
        n0, n1 = nodes[i], nodes[i + 1]
        lat0, lon0 = latlon_sequence[i]
        lat1, lon1 = latlon_sequence[i + 1]
        _, _, straight_m = geod.inv(lon0, lat0, lon1, lat1)

        if n0 == n1 or straight_m < 30:
            full_coords.append((lat1, lon1))
            continue

        try:
            path = nx.shortest_path(G, n0, n1, weight="length")
            path_len_m = sum(
                G.edges[path[k], path[k + 1], 0].get("length", 0)
                for k in range(len(path) - 1)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            full_coords.append((lat1, lon1))
            continue

        if straight_m > 0 and path_len_m > straight_m * max_detour_ratio:
            # shortest path가 우회 너무 큼 → 직선 fallback
            full_coords.append((lat1, lon1))
            continue

        # path 따라 좌표 추가 (시작점은 이미 들어가 있음)
        for node in path[1:]:
            pt = G.nodes[node]
            coord = (pt["y"], pt["x"])
            if full_coords[-1] != coord:
                full_coords.append(coord)
        # 마지막 정거장 원본 좌표로 끝맺기 (스냅 오차 보정)
        if full_coords[-1] != (lat1, lon1):
            full_coords.append((lat1, lon1))

    return full_coords


def export_arterial_to_geojson():
    """주간선도로 그래프를 GeoJSON으로 저장 (지도 레이어용)"""
    from src.config import DATA_GEOJSON
    G = get_arterial_graph()
    edges_gdf = ox.graph_to_gdfs(G, nodes=False).to_crs("EPSG:4326")
    keep_cols = [c for c in ["highway", "name", "lanes"] if c in edges_gdf.columns]
    edges_gdf = edges_gdf[keep_cols + ["geometry"]].copy()
    # 리스트 컬럼은 GeoJSON 직렬화 실패하므로 문자열화
    for col in keep_cols:
        edges_gdf[col] = edges_gdf[col].apply(
            lambda v: ", ".join(v) if isinstance(v, list) else (v if v is not None else "")
        )
    out = DATA_GEOJSON / "daejeon_arterial.geojson"
    edges_gdf.to_file(out, driver="GeoJSON")
    print(f"✅ 주간선도로 GeoJSON: {out} ({len(edges_gdf)} 엣지)")
    return out


def export_arterial_to_shp():
    """주간선도로를 SHP로 저장 (외부 GIS 도구용)"""
    from src.config import DATA_PROCESSED
    G = get_arterial_graph()
    edges_gdf = ox.graph_to_gdfs(G, nodes=False).to_crs("EPSG:4326")
    # SHP 호환 정리
    keep_cols = [c for c in ["highway", "name"] if c in edges_gdf.columns]
    edges_gdf = edges_gdf[keep_cols + ["geometry"]].copy()
    for col in keep_cols:
        edges_gdf[col] = edges_gdf[col].apply(
            lambda v: ", ".join(v) if isinstance(v, list) else (str(v) if v is not None else "")
        )
    out = DATA_PROCESSED / "daejeon_arterial.shp"
    edges_gdf.to_file(out, driver="ESRI Shapefile", encoding="utf-8")
    print(f"✅ 주간선도로 SHP: {out}")
    return out


if __name__ == "__main__":
    G = get_road_graph()
    print(f"노드 수: {G.number_of_nodes():,}")
    print(f"엣지 수: {G.number_of_edges():,}")
    # 빠른 테스트: 정부청사역 → 국립중앙과학관
    test = snap_path_to_road([(36.358, 127.383), (36.383, 127.378)], G=G)
    print(f"테스트 경로 좌표 수: {len(test)}")
