import geopandas as gpd

HYDROLAKES_CRS = "EPSG:4326"  # HydroLAKES is distributed in WGS84


def read_lakes(cfg, target_crs=None):
    """Read HydroLAKES; assume WGS84 if the .prj is missing, then reproject."""
    lakes = gpd.read_file(cfg.lakes_path)
    if lakes.crs is None:
        lakes = lakes.set_crs(HYDROLAKES_CRS)
    if target_crs is not None and lakes.crs != target_crs:
        lakes = lakes.to_crs(target_crs)
    return lakes
