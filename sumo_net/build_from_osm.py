"""SUMO network builder for Sector-78 -> Vasundhara corridor.

Steps:
1. Download an OSM extract for the configured bounding box.
2. Run netconvert + randomTrips + duarouter to build baseline network/routes.

This is a lightweight scaffold; real deployments should include robust caching,
validation, and better trip generation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
OSM_FILE = Path(__file__).parent / "corridor.osm.xml"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def download_osm(config):
    bbox = config["region"]["bbox"]
    query = f"[out:xml];way({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']})[highway];(._;>;);out;"                                                 
    cmd = [
        "curl",
        "-o",
        str(OSM_FILE),
        "-d",
        f"data={query}",
        config["region"].get("osm_extract_url", "https://overpass-api.de/api/interpreter"),
    ]
    subprocess.run(cmd, check=True)


def build_sumo_files(config):
    sumo_home = Path.getenv("SUMO_HOME")  # type: ignore[attr-defined]
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME not set")
    netconvert = Path(sumo_home) / "bin" / "netconvert"
    random_trips = Path(sumo_home) / "tools" / "randomTrips.py"
    output_net = Path(__file__).parent / "corridor.net.xml"
    subprocess.run([str(netconvert), "--osm-files", str(OSM_FILE), "-o", str(output_net)], check=True)
    subprocess.run([
        "python",
        str(random_trips),
        "-n",
        str(output_net),
        "-o",
        str(Path(__file__).parent / "corridor.rou.xml"),
        "-b",
        "0",
        "-e",
        "600",
        "-p",
        "1",
    ], check=True)


def main():
    config = load_config()
    download_osm(config)
    build_sumo_files(config)


if __name__ == "__main__":
    main()
