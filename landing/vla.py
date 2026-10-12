"""SmolVLA adaptation, training and local inference; run in environments/vla."""

import argparse
import base64
import io
import json
import math
import random
import time
import uuid
from pathlib import Path

from .core import (
    ACTIONS,
    FEATURES,
    LIMITS,
    clamp_actions,
    digest,
    read_json,
    scenario_hash,
)
from .data import training_episodes as training_episodes
from .storage import atomic_json, file_digest
from .vision import INSTRUCTION, STATE_SCALE

ROOT = Path(__file__).resolve().parents[1]
IMAGE_KEY = "observation.images.camera1"


def snapshots(lock):
    from huggingface_hub import snapshot_download

    paths = {}
    for key in ("policy", "backbone"):
        patterns = ["*.json", "*.txt", "*.model"]
        if key == "policy":
            patterns += ["*.safetensors"]
        paths[key] = snapshot_download(
            **lock[key], allow_patterns=patterns, cache_dir=ROOT / "runs/hf-cache"
        )
    return paths


def load_policy(bundle, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers import AutoTokenizer

    status_path = Path(bundle) / "status.json"
    if status_path.exists() and read_json(status_path).get("status") != "complete":
        raise ValueError("Training bundle is incomplete")
    manifest = read_json(Path(bundle) / "landing.json")
    if (
        manifest["features"] != FEATURES
        or manifest["actions"] != ACTIONS
        or manifest["state_scale"] != STATE_SCALE
    ):
        raise ValueError("Unsupported landing policy schema")
    paths = snapshots(manifest["model_lock"])
    config = PreTrainedConfig.from_pretrained(bundle)
    config.device = device
    config.vlm_model_name = paths["backbone"]
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(bundle, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(paths["backbone"])
    policy.eval()
    return policy, tokenizer, manifest


def prepare(args):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if args.output.exists():
        raise ValueError("Output exists")
    lock = read_json(ROOT / "config/vla-model-lock.json")
    paths = snapshots(lock)
    config = PreTrainedConfig.from_pretrained(paths["policy"])
    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(len(FEATURES),)
        ),
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(len(ACTIONS),))
    }
    config.normalization_mapping = {
        key: NormalizationMode.IDENTITY for key in ("VISUAL", "STATE", "ACTION")
    }
    config.vlm_model_name = paths["backbone"]
    config.load_vlm_weights = False
    config.device = args.device
    config.chunk_size = config.n_action_steps = 1
    config.resize_imgs_with_padding = (256, 256)
    config.push_to_hub = False
    policy = SmolVLAPolicy.from_pretrained(paths["policy"], config=config).to(
        args.device
    )
    scenario = read_json(args.scenario)
    manifest = {
        "schema": 1,
        "stage": "base_adapted",
        "model_lock": lock,
        "scenario": scenario,
        "scenario_hash": scenario_hash(scenario),
        "instruction": args.instruction,
        "features": FEATURES,
        "actions": ACTIONS,
        "state_scale": STATE_SCALE,
        "image_size": [256, 256],
        "action_encoding": "linear_to_minus1_plus1",
        "note": "Pretrained robot weights mapped to aircraft channels; not a pretrained flight controller.",
    }
    args.output.mkdir(parents=True)
    policy.save_pretrained(args.output)
    atomic_json(args.output / "landing.json", manifest)
    print("Prepared adapted baseline:", args.output)


def state_normalization(manifest):
    """Offset and scale for the state vector.

    Fine-tuned models store the training data's mean and standard deviation, so a
    glide path deviation of tens of feet is visible next to kilometres of distance.
    Earlier models (and the untrained baseline) use the fixed STATE_SCALE.
    """
    if "state_mean" in manifest:
        return manifest["state_mean"], manifest["state_std"]
    return [0.0] * len(FEATURES), STATE_SCALE


def batch_input(
    tokenizer, observations, images, instruction, device, actions=None, norm=None
):
    import numpy as np
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from PIL import ImageOps

    if any(
        len(row) != len(FEATURES) or not all(math.isfinite(v) for v in row)
        for row in observations
    ):
        raise ValueError("Invalid observation vector")
    if (
        not images
        or len(observations) != len(images)
        or (actions is not None and len(actions) != len(images))
    ):
        raise ValueError("Mismatched or empty input batch")
    if actions is not None:
        for action in actions:
            if len(action) != len(ACTIONS) or any(
                not math.isfinite(v) or not lo <= v <= hi
                for v, (lo, hi) in zip(action, LIMITS)
            ):
                raise ValueError("Invalid training action")
    pixels = [
        np.asarray(
            ImageOps.pad(im.convert("RGB"), (256, 256)), dtype=np.float32
        ).transpose(2, 0, 1)
        / 255
        for im in images
    ]
    tokens = tokenizer(
        [instruction + "\n"] * len(images),
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    batch = {
        "observation.state": torch.tensor(
            [[(v - m) / s for v, m, s in zip(row, *norm)] for row in observations],
            dtype=torch.float32,
        ),
        IMAGE_KEY: torch.tensor(np.stack(pixels)),
        OBS_LANGUAGE_TOKENS: tokens["input_ids"],
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].bool(),
    }
    if actions is not None:
        batch["action"] = torch.tensor(
            [
                [[2 * (v - lo) / (hi - lo) - 1 for v, (lo, hi) in zip(row, LIMITS)]]
                for row in actions
            ],
            dtype=torch.float32,
        )
    return {key: value.to(device) for key, value in batch.items()}


def decode_action(values):
    if len(values) != len(ACTIONS):
        raise ValueError("Wrong action dimension")
    return clamp_actions(
        [lo + (v + 1) * 0.5 * (hi - lo) for v, (lo, hi) in zip(values, LIMITS)]
    )


def predict(policy, tokenizer, manifest, observations, images, device):
    import torch

    batch = batch_input(
        tokenizer,
        observations,
        images,
        manifest["instruction"],
        device,
        norm=state_normalization(manifest),
    )
    with torch.inference_mode():
        action = policy.select_action(batch).cpu().tolist()
    return [decode_action(row) for row in action]


def train(args):
    import torch
    from PIL import Image

    manifest = read_json(args.base / "landing.json")
    episodes = training_episodes(args.episodes, manifest)
    if args.output.exists():
        raise ValueError("Output exists")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    nval = max(1, round(len(episodes) * 0.2))
    validation, fitting = episodes[:nval], episodes[nval:]
    samples = [s for ep in fitting for s in ep["samples"]]
    val_samples = [s for ep in validation for s in ep["samples"]]
    policy, tokenizer, manifest = load_policy(args.base, args.device)
    obs = [s["observation"] for s in samples]
    mean = [sum(col) / len(col) for col in zip(*obs)]
    std = [
        max((sum((v - m) ** 2 for v in col) / len(col)) ** 0.5, 1e-3)
        for col, m in zip(zip(*obs), mean)
    ]
    norm = (mean, std)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr
    )
    history = []
    args.output.mkdir(parents=True)
    atomic_json(args.output / "status.json", {"status": "training"})

    def make_batch(items):
        images = []
        for item in items:
            with Image.open(item["image"]) as im:
                images.append(im.convert("RGB"))
        return batch_input(
            tokenizer,
            [s["observation"] for s in items],
            images,
            manifest["instruction"],
            args.device,
            [s["action"] for s in items],
            norm,
        )

    for step in range(args.steps):
        policy.train()
        batch = make_batch(rng.choices(samples, k=args.batch_size))
        loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            policy.eval()
            # Fixed validation subset and flow-matching noise seed make checkpoints comparable.
            with torch.random.fork_rng(
                devices=[torch.cuda.current_device()] if args.device == "cuda" else []
            ):
                torch.manual_seed(args.seed)
                with torch.no_grad():
                    val_loss, _ = policy(make_batch(val_samples[: args.batch_size]))
            result = {
                "step": step + 1,
                "train_loss": loss.item(),
                "validation_subset_loss": val_loss.item(),
            }
            history.append(result)
            print(json.dumps(result), flush=True)
    policy.save_pretrained(args.output)
    manifest = {
        **manifest,
        "stage": "finetuned",
        "state_mean": mean,
        "state_std": std,
        "parent_manifest_hash": digest(manifest),
        "parent_weights_sha256": file_digest(args.base / "model.safetensors"),
    }
    atomic_json(args.output / "landing.json", manifest)
    atomic_json(
        args.output / "training.json",
        {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "history": history,
            "train_episodes": [
                {k: v for k, v in ep.items() if k != "samples"} for ep in fitting
            ],
            "validation_episodes": [
                {k: v for k, v in ep.items() if k != "samples"} for ep in validation
            ],
            "train_samples": len(samples),
            "validation_samples": len(val_samples),
            "note": "Final checkpoint; validation loss is a fixed subset diagnostic, not landing success.",
        },
    )
    atomic_json(args.output / "status.json", {"status": "complete"})


def serve(args):
    import torch
    from PIL import Image

    if args.bridge.exists():
        raise ValueError("Use a new bridge directory for each worker")
    policy, tokenizer, manifest = load_policy(args.model, args.device)
    manifest = {
        **manifest,
        "weights_sha256": file_digest(args.model / "model.safetensors"),
        "seed": args.seed,
    }
    args.bridge.mkdir(parents=True)
    worker = uuid.uuid4().hex
    atomic_json(args.bridge / "ready.json", {"worker_id": worker, "manifest": manifest})
    print("VLA worker ready:", args.bridge, flush=True)
    session = None
    try:
        while True:
            for path in sorted(args.bridge.glob("*.request.json")):
                result = {"worker_id": worker}
                try:
                    request = read_json(path)
                    if (
                        request["worker_id"] != worker
                        or request["scenario_hash"] != manifest["scenario_hash"]
                    ):
                        raise ValueError("Request targets another worker/scenario")
                    if request["session_id"] != session:
                        session = request["session_id"]
                        policy.reset()
                        torch.manual_seed(args.seed)
                    data = base64.b64decode(request["image"], validate=True)
                    with Image.open(io.BytesIO(data)) as im:
                        result["action"] = predict(
                            policy,
                            tokenizer,
                            manifest,
                            [request["features"]],
                            [im.convert("RGB")],
                            args.device,
                        )[0]
                except Exception as exc:
                    result["error"] = str(exc)
                if path.exists():
                    atomic_json(
                        path.with_name(
                            path.name.replace(".request.json", ".response.json")
                        ),
                        result,
                    )
                    path.unlink(missing_ok=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        (args.bridge / "ready.json").unlink(missing_ok=True)


def check(args):
    import torch
    from PIL import Image

    policy, tokenizer, manifest = load_policy(args.model, args.device)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    observation = [0.0] * len(FEATURES)
    result = predict(
        policy,
        tokenizer,
        manifest,
        [observation],
        [Image.new("RGB", (256, 256))],
        args.device,
    )
    report = {
        "input": "synthetic black image and zero state; not a flight result",
        "action": result[0],
        "inference_seconds": time.monotonic() - started,
        "device": args.device,
    }
    if args.backward:
        policy.train()
        batch = batch_input(
            tokenizer,
            [observation],
            [Image.new("RGB", (256, 256))],
            manifest["instruction"],
            args.device,
            [[0, 0, 0, 0.3]],
            state_normalization(manifest),
        )
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=1e-4
        )
        loss, _ = policy(batch)
        loss.backward()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite smoke-test loss")
        report["backward_loss"] = loss.item()
        report["finite_gradients"] = all(
            torch.isfinite(p.grad).all().item()
            for p in policy.parameters()
            if p.grad is not None
        )
        if not report["finite_gradients"]:
            raise ValueError("Non-finite gradients")
        optimizer.step()
        report["optimizer_step"] = "passed; updated weights not saved"
    if args.device == "cuda":
        report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--scenario", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--instruction", default=INSTRUCTION)
    q = sub.add_parser("train")
    q.add_argument("episodes", nargs="+", type=Path)
    q.add_argument("--base", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--steps", type=int, default=1000)
    q.add_argument("--batch-size", type=int, default=1)
    q.add_argument("--lr", type=float, default=1e-4)
    q.add_argument("--log-every", type=int, default=50)
    q = sub.add_parser("serve")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--bridge", type=Path, required=True)
    q = sub.add_parser("check")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--backward", action="store_true")
    for q in sub.choices.values():
        q.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        q.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.command == "train" and (
        a.steps < 1 or a.batch_size < 1 or a.log_every < 1 or not 0 < a.lr < 1
    ):
        p.error("Invalid training parameters")
    globals()[a.command](a)


if __name__ == "__main__":
    main()
