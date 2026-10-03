"""Temporary synthetic integration test; never creates human demonstrations."""

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landing import vla
from landing.core import FEATURES, read_json


def main():
    import torch
    from PIL import Image

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="landing-vla-test-") as tmp:
        root = Path(tmp)
        image = root / "synthetic.jpg"
        Image.new("RGB", (256, 256), (40, 80, 120)).save(image)
        episodes = [
            {
                "path": f"synthetic-test-{i}",
                "hash": str(i),
                "view": 0,
                "samples": [
                    {
                        "observation": [i * 0.01] * len(FEATURES),
                        "image": str(image),
                        "action": [0, 0, 0, 0.3, 0, 0, 0],
                    }
                ],
            }
            for i in range(3)
        ]
        args = argparse.Namespace(
            base=a.base,
            episodes=[],
            output=root / "model",
            seed=42,
            device="cuda",
            steps=1,
            batch_size=1,
            lr=1e-4,
            log_every=1,
        )
        # Dataset admission is tested separately. Synthetic fixtures cannot pass as human flights.
        with patch.object(vla, "training_episodes", return_value=episodes):
            vla.train(args)
        assert read_json(args.output / "status.json")["status"] == "complete"
        report = read_json(args.output / "training.json")
        assert not (
            {r["hash"] for r in report["train_episodes"]}
            & {r["hash"] for r in report["validation_episodes"]}
        )
        gc.collect()
        torch.cuda.empty_cache()
        vla.check(
            argparse.Namespace(
                model=args.output, device="cuda", seed=42, backward=False
            )
        )
        print(
            json.dumps(
                {
                    "synthetic_train_save_reload": "passed",
                    "persisted_model": False,
                    "not_a_flight_result": True,
                }
            )
        )


if __name__ == "__main__":
    main()
