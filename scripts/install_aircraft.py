"""Fetch pinned aircraft with Git into a user-selected directory (WSL)."""
import argparse
import json
from pathlib import Path
import subprocess


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def apply_patch(directory, item):
    patch_name = item.get("patch")
    expected = ""
    if patch_name:
        patch = Path(__file__).resolve().parents[1] / patch_name
        expected = patch.read_text().strip()
    actual = git(directory, "diff", "--binary")
    if git(directory, "diff", "--cached") or git(directory, "ls-files", "--others", "--exclude-standard"):
        raise RuntimeError(f"Unexpected local changes: {directory}")
    if actual:
        if actual != expected:
            raise RuntimeError(f"Unexpected local changes: {directory}")
    elif expected:
        git(directory, "apply", "--check", str(patch))
        git(directory, "apply", str(patch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "config/aircraft-lock.json").read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, item in manifest.items():
        dest = args.destination / item["directory"]
        if dest.exists():
            if git(dest, "rev-parse", "HEAD") != item["commit"]:
                raise RuntimeError(f"Existing checkout differs; preserve and inspect it: {dest}")
            apply_patch(dest, item)
            print(f"Already installed: {name}", flush=True)
            continue
        dest.mkdir()
        git(dest, "init")
        git(dest, "remote", "add", "origin", item["repository"])
        if item["sparse_path"]:
            git(dest, "config", "remote.origin.promisor", "true")
            git(dest, "config", "remote.origin.partialclonefilter", "blob:none")
            git(dest, "sparse-checkout", "init", "--cone")
            git(dest, "sparse-checkout", "set", item["sparse_path"])
        print(f"Fetching {name} at {item['commit']}", flush=True)
        git(dest, "fetch", "--depth=1", "--filter=blob:none", "origin", item["commit"])
        git(dest, "checkout", "--detach", "FETCH_HEAD")
        apply_patch(dest, item)
        print(f"Installed: {dest}", flush=True)


if __name__ == "__main__":
    main()
