#!/usr/bin/env python3
"""Download and standardize continuous multifacility location benchmarks.

The script creates one CSV per instance with columns ``id,x,y,weight`` and a
manifest describing provenance.  It also creates ``experiments.csv``, the
Cartesian product of instances, facility counts, and supported norm orders.

Sources
-------
* Eilon--Watson-Gandy--Christofides instances and location datasets archived
  by vblancoOR/mclpif.
* TSPLIB p654 and u1060, both used as large multisource Weber benchmarks.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GITHUB_API = "https://api.github.com/repos/vblancoOR/mclpif/contents"
RAW_ROOT = "https://raw.githubusercontent.com/vblancoOR/mclpif/master/instances"
USER_AGENT = "sdp-clusters-benchmark-fetcher/1.0"


@dataclass(frozen=True)
class InstanceSource:
    name: str
    family: str
    url: str
    citation: str
    parser: str
    fallback_url: str = ""


OPEN_LOCATION_FILES = (
    "AP200.txt",
    "coord324.txt",
    "coord402.txt",
    "coord500.txt",
    "coord818.txt",
    "ch2863.txt",
)


TSPLIB_SOURCES = (
    InstanceSource(
        "p654",
        "tsplib",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/p654.tsp.gz",
        "Reinelt (1991), TSPLIB; used in multisource Weber experiments",
        "tsplib",
        "https://raw.githubusercontent.com/mastqe/tsplib/master/p654.tsp",
    ),
    InstanceSource(
        "u1060",
        "tsplib",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/u1060.tsp.gz",
        "Reinelt (1991), TSPLIB; used in multisource Weber experiments",
        "tsplib",
        "https://raw.githubusercontent.com/mastqe/tsplib/master/u1060.tsp",
    ),
)


def _request(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        encoding = response.headers.get("Content-Encoding", "").lower()
    if url.endswith(".gz") and encoding != "gzip":
        payload = gzip.decompress(payload)
    return payload


def _download(source: InstanceSource, timeout: float) -> bytes:
    try:
        return _request(source.url, timeout)
    except (OSError, urllib.error.URLError, gzip.BadGzipFile):
        if not source.fallback_url:
            raise
        return _request(source.fallback_url, timeout)


def _eilon_sources(timeout: float) -> list[InstanceSource]:
    folder = urllib.parse.quote("instances/Eilon et al", safe="/")
    listing = json.loads(_request(f"{GITHUB_API}/{folder}?ref=master", timeout))
    sources = []
    for item in listing:
        name = item["name"]
        if item.get("type") == "file" and re.fullmatch(r"eilon\d+(?:_\d+)?\.csv", name):
            sources.append(
                InstanceSource(
                    name[:-4],
                    "eilon",
                    item["download_url"],
                    "Eilon, Watson-Gandy, and Christofides (1971)",
                    "tabular",
                )
            )
    return sorted(sources, key=lambda source: (instance_size(source.name), source.name))


def _open_location_sources() -> list[InstanceSource]:
    citations = {
        "AP200.txt": "Ernst and Krishnamoorthy (1996)",
        "ch2863.txt": "Taillard (2003), centroid clustering benchmark",
    }
    return [
        InstanceSource(
            Path(filename).stem,
            "location_archive",
            f"{RAW_ROOT}/{filename}",
            citations.get(filename, "Senne, Pereira, and Lorena (2010)"),
            "tabular",
        )
        for filename in OPEN_LOCATION_FILES
    ]


def instance_size(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def _parse_tabular(payload: bytes) -> list[tuple[float, float, float]]:
    text = payload.decode("utf-8-sig")
    rows = []
    if "," in text.splitlines()[0]:
        reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
        for row in reader:
            clean = {str(key).strip().lower(): str(value).strip() for key, value in row.items()}
            x = float(clean["longitude"])
            y = float(clean["latitude"])
            weight = float(clean.get("weight", "1") or 1.0)
            rows.append((x, y, weight))
    else:
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 2:
                rows.append((float(fields[0]), float(fields[1]), float(fields[2]) if len(fields) > 2 else 1.0))
    return rows


def _parse_tsplib(payload: bytes) -> list[tuple[float, float, float]]:
    text = payload.decode("ascii", errors="strict")
    rows: list[tuple[float, float, float]] = []
    in_coordinates = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "NODE_COORD_SECTION":
            in_coordinates = True
            continue
        if not in_coordinates:
            continue
        if line in {"EOF", "DISPLAY_DATA_SECTION"}:
            break
        fields = line.split()
        if len(fields) >= 3:
            rows.append((float(fields[1]), float(fields[2]), 1.0))
    if not rows:
        raise ValueError("TSPLIB file has no NODE_COORD_SECTION")
    return rows


def _normalize(rows: Iterable[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    rows = list(rows)
    x_values = [row[0] for row in rows]
    y_values = [row[1] for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_scale = x_max - x_min or 1.0
    y_scale = y_max - y_min or 1.0
    return [((x - x_min) / x_scale, (y - y_min) / y_scale, weight) for x, y, weight in rows]


def _write_instance(path: Path, rows: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "x", "y", "weight"))
        for index, (x, y, weight) in enumerate(rows, start=1):
            writer.writerow((index, f"{x:.12g}", f"{y:.12g}", f"{weight:.12g}"))


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    fields = ("instance", "family", "n", "dimension", "weighted", "file", "source_url", "citation", "sha256")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_experiments(
    path: Path,
    records: list[dict[str, object]],
    facilities: list[int],
    norms: list[str],
) -> None:
    fields = ("experiment_id", "instance", "instance_file", "n", "p", "tau")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        experiment = 0
        for record in records:
            for number in facilities:
                if number >= int(record["n"]):
                    continue
                for norm in norms:
                    experiment += 1
                    writer.writerow(
                        {
                            "experiment_id": experiment,
                            "instance": record["instance"],
                            "instance_file": record["file"],
                            "n": record["n"],
                            "p": number,
                            "tau": norm,
                        }
                    )


def fetch_instances(
    output: Path,
    families: set[str],
    facilities: list[int],
    norms: list[str],
    normalize: bool,
    timeout: float,
    overwrite: bool,
) -> list[dict[str, object]]:
    """Download selected families and return their manifest records."""

    sources: list[InstanceSource] = []
    if "eilon" in families:
        sources.extend(_eilon_sources(timeout))
    if "location_archive" in families:
        sources.extend(_open_location_sources())
    if "tsplib" in families:
        sources.extend(TSPLIB_SOURCES)
    output.mkdir(parents=True, exist_ok=True)
    processed = output / "processed"
    raw = output / "raw"
    records: list[dict[str, object]] = []
    for source in sources:
        payload = _download(source, timeout)
        suffix = ".tsp" if source.parser == "tsplib" else Path(urllib.parse.urlparse(source.url).path).suffix
        raw_path = raw / source.family / f"{source.name}{suffix}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not raw_path.exists():
            raw_path.write_bytes(payload)
        rows = _parse_tsplib(payload) if source.parser == "tsplib" else _parse_tabular(payload)
        if normalize:
            rows = _normalize(rows)
        destination = processed / source.family / f"{source.name}.csv"
        if overwrite or not destination.exists():
            _write_instance(destination, rows)
        relative = destination.relative_to(output).as_posix()
        records.append(
            {
                "instance": source.name,
                "family": source.family,
                "n": len(rows),
                "dimension": 2,
                "weighted": int(any(not math.isclose(row[2], 1.0) for row in rows)),
                "file": relative,
                "source_url": source.url,
                "citation": source.citation,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        print(f"{source.name:14s} n={len(rows):5d} -> {relative}")
    records.sort(key=lambda row: (int(row["n"]), str(row["instance"])))
    _write_manifest(output / "manifest.csv", records)
    _write_experiments(output / "experiments.csv", records, facilities, norms)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("instances"))
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("eilon", "location_archive", "tsplib"),
        default=("eilon", "location_archive", "tsplib"),
    )
    parser.add_argument("--facilities", nargs="+", type=int, default=(2, 3, 5, 10))
    parser.add_argument("--norms", nargs="+", default=("1", "1.5", "2", "3", "inf"))
    parser.add_argument("--keep-original-scale", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.facilities):
        parser.error("facility counts must be positive")
    fetch_instances(
        arguments.output,
        set(arguments.families),
        list(arguments.facilities),
        list(arguments.norms),
        not arguments.keep_original_scale,
        arguments.timeout,
        arguments.overwrite,
    )


if __name__ == "__main__":
    main()
