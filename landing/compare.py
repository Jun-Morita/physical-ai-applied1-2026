"""Compare all specified before/after VLA flights without hiding failed runs."""

import argparse
import json
from pathlib import Path

from .core import digest, evaluate, read_json
from .storage import rows_at


def compare(before, after):
    if not before or not after:
        raise ValueError("Both comparison groups require flights")
    paths = [Path(p).resolve() for p in list(before) + list(after)]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate flight paths")
    condition = None
    groups = {}
    before_weights = None
    for name, paths, stage in [
        ("before", before, "base_adapted"),
        ("after", after, "finetuned"),
    ]:
        flights = []
        hashes = set()
        for path in paths:
            meta = read_json(Path(path) / "metadata.json")
            if meta.get("purpose", "evaluation") != "evaluation":
                raise ValueError("Diagnostic episodes cannot be scored")
            policy = meta.get("policy_manifest") or {}
            if meta.get("source") != "ai" or policy.get("stage") != stage:
                raise ValueError(f"{path}: wrong policy stage or source")
            if stage == "base_adapted":
                before_weights = policy["weights_sha256"]
            elif policy.get("parent_weights_sha256") != before_weights:
                raise ValueError(
                    "Finetuned model is not derived from the compared baseline"
                )
            key = digest(
                [
                    meta["scenario"],
                    meta["instruction"],
                    meta["execution_mode"],
                    meta["image_period"],
                    meta.get("requested_seconds"),
                    policy["seed"],
                ]
            )
            if condition is not None and key != condition:
                raise ValueError("Evaluation conditions differ")
            condition = key
            hashes.add(meta["policy_hash"])
            rows = rows_at(path) if (Path(path) / "samples.jsonl").exists() else []
            metrics = evaluate(rows, meta["scenario"])
            flights.append(
                {
                    "path": str(path),
                    "status": meta["status"],
                    "error": meta.get("error"),
                    "candidate_success": meta["status"] == "complete"
                    and metrics["candidate_success"],
                    "metrics": metrics,
                }
            )
        if len(hashes) != 1:
            raise ValueError("Use exactly one model per comparison group")
        groups[name] = {
            "flights": flights,
            "count": len(flights),
            "candidate_success_rate": sum(f["candidate_success"] for f in flights)
            / len(flights),
        }
    return {
        "condition_hash": condition,
        **groups,
        "note": "Candidate metrics, not reviewed landing success. Include all attempts; repeated identical starts do not measure generalization.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", nargs="+", type=Path, required=True)
    p.add_argument("--after", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if len(set(path.resolve() for path in a.before + a.after)) != len(
        a.before + a.after
    ):
        p.error("Duplicate flight paths")
    result = compare(a.before, a.after)
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
