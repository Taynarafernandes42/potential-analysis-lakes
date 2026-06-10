import argparse
from pathlib import Path

import geopandas as gpd


def load_lake_metadata(shapefile_path):
    lakes = gpd.read_file(shapefile_path)
    lakes["Hylak_id"] = lakes["Hylak_id"].astype(int)
    lakes = lakes.set_index("Hylak_id")

    metadata = {}
    for hid, row in lakes.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            centroid = (None, None)
        else:
            point = geom.representative_point()
            centroid = (point.x, point.y)

        lake_name = row.get("Lake_name")
        metadata[hid] = {
            "name": lake_name if lake_name not in (None, "") else None,
            "centroid": centroid,
        }
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Lookup HydroLAKES lake names and centroid coordinates by Hylak_id "
            "from the Germany shapefile."
        )
    )
    parser.add_argument(
        "ids",
        nargs="+",
        type=int,
        help="One or more Hylak_id values to look up."
    )
    parser.add_argument(
        "--shapefile",
        default="Data/Raw/HydroLAKES_germany.shp",
        help="Path to the HydroLAKES shapefile."
    )
    args = parser.parse_args()

    shapefile_path = Path(args.shapefile)
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    lake_metadata = load_lake_metadata(shapefile_path)

    for hid in args.ids:
        info = lake_metadata.get(hid)
        if info is None:
            print(f"{hid}: <not found>")
            continue

        name = info["name"] or "<no name available>"
        x, y = info["centroid"]
        if x is None or y is None:
            coords = "<no centroid available>"
        else:
            coords = f"{y:.6f} {x:.6f}"

        print(f"{hid}: {name} (centroid: {coords})")


if __name__ == "__main__":
    main()
