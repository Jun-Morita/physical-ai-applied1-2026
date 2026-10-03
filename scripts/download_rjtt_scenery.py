"""Download the RJTT-area TerraSync tile with upstream SHA-1 verification."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

BASE = "https://terrasync.b-cdn.net/"
REGIONS = ("Terrain/e130n30/e139n35", "Objects/e130n30/e139n35", "Airports/R/J/T")


def read(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return response.read()


def inventory(folder):
    index = read(folder + "/.dirindex").decode()
    entries = []
    for line in index.splitlines():
        fields = line.split(":")
        if fields[0] not in ("f", "d"):
            continue
        name = fields[1]
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError(f"Invalid entry: {name}")
        path = folder + "/" + name
        if fields[0] == "d":
            entries.extend(inventory(path))
        else:
            entries.append({"path": path, "sha1": fields[2], "size": int(fields[3])})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    entries = [entry for folder in REGIONS for entry in inventory(folder)]
    print(
        f"Downloading {len(entries)} files ({sum(e['size'] for e in entries) / 1024**2:.1f} MiB)",
        flush=True,
    )

    def fetch(entry):
        path = args.destination.joinpath(*PurePosixPath(entry["path"]).parts)
        if (
            path.exists()
            and hashlib.sha1(path.read_bytes()).hexdigest() == entry["sha1"]
        ):
            return
        data = read(entry["path"])
        if (
            len(data) != entry["size"]
            or hashlib.sha1(data).hexdigest() != entry["sha1"]
        ):
            raise RuntimeError(f"Checksum mismatch: {entry['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(data)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, entries))
    manifest = {
        "source": BASE,
        "regions": REGIONS,
        "files": entries,
        "note": "Current upstream snapshot; referenced shared Models are not included.",
    }
    (args.destination / "physical-ai-scenery-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        "Scenery files verified. Shared Models may still require TerraSync.", flush=True
    )


if __name__ == "__main__":
    main()
