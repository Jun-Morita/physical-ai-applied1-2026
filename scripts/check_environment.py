"""Smoke-check CUDA training and the bundled JSBSim physics engine."""

import json
import math
import platform

import jsbsim
import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the Windows NVIDIA driver and WSL GPU access")
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(32, 4, device="cuda")
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite CUDA training loss")

    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    # The bundled JSBSim C172P is an engine smoke test, not validation of FlightGear C172P.
    if not fdm.load_model("c172p"):
        raise RuntimeError("Unable to load bundled JSBSim smoke-test aircraft")
    fdm.set_dt(1 / 120)
    fdm["ic/h-sl-ft"] = 3000
    fdm["ic/vc-kts"] = 90
    if not fdm.run_ic():
        raise RuntimeError("JSBSim initialization failed")
    for _ in range(120):
        if not fdm.run():
            raise RuntimeError("JSBSim step failed")
    altitude = fdm["position/h-sl-ft"]
    if not math.isfinite(altitude) or not math.isclose(fdm.get_sim_time(), 1.0):
        raise RuntimeError("Invalid JSBSim simulation result")
    print(json.dumps({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "cuda_training_step": "passed",
        "jsbsim": jsbsim.__version__,
        "jsbsim_simulated_seconds": fdm.get_sim_time(),
        "aircraft_validation": "FlightGear C172P, 747-400 and F-14B flight performance not validated by this check",
    }, indent=2))


if __name__ == "__main__":
    main()
