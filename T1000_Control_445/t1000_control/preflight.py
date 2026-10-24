from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_delivery(root: Path | str | None = None) -> dict:
    package_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    meta = settings.POLICY_REGISTRY.get(settings.DEFAULT_POLICY)
    errors: list[str] = settings.validate_settings()
    report: dict = {
        "root": str(package_root.resolve()),
        "offline_only": True,
        "policy": {"name": settings.DEFAULT_POLICY},
        "errors": errors,
    }

    if errors:
        report["ok"] = False
        return report

    if meta is None:
        errors.append(f"DEFAULT_POLICY {settings.DEFAULT_POLICY!r} is not registered")
        report["ok"] = False
        return report

    model_path = package_root / "models" / settings.DEFAULT_POLICY
    target = report["policy"]
    target["path"] = str(model_path)
    target["exists"] = model_path.is_file()
    if not model_path.is_file():
        target["sha256_ok"] = False
        errors.append(f"missing policy: {model_path}")
    else:
        actual = _sha256(model_path)
        expected = meta["onnx_sha256"]
        target["sha256"] = actual
        target["sha256_ok"] = actual == expected
        if actual != expected:
            errors.append(f"policy SHA-256 mismatch: {actual} != {expected}")

    if model_path.is_file():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                errors.append(
                    f"expected one ONNX input/output, got {len(inputs)}/{len(outputs)}"
                )
            else:
                model_input, model_output = inputs[0], outputs[0]
                report["policy"]["input"] = {
                    "name": model_input.name,
                    "shape": list(model_input.shape),
                }
                report["policy"]["output"] = {
                    "name": model_output.name,
                    "shape": list(model_output.shape),
                }
                if model_input.name != "obs" or model_input.shape[-1] != settings.OBS_DIM:
                    errors.append(f"unexpected ONNX input: {model_input.name}{model_input.shape}")
                if model_output.name != "action" or model_output.shape[-1] != settings.ACT_DIM:
                    errors.append(f"unexpected ONNX output: {model_output.name}{model_output.shape}")

                observation = np.linspace(
                    -1.0, 1.0, settings.OBS_DIM, dtype=np.float32
                )[None, :]
                action = session.run(None, {model_input.name: observation})[0][0]
                if action.shape != (settings.ACT_DIM,) or not np.isfinite(action).all():
                    errors.append(f"invalid sample action: shape={action.shape}, value={action}")
                report["policy"]["sample_action"] = [float(value) for value in action]
                report["policy"]["provider"] = session.get_providers()[0]
        except ImportError:
            errors.append("onnxruntime is not installed; install with: pip install -e '.[inference]'")
        except Exception as exc:
            errors.append(f"ONNX load/inference failed: {type(exc).__name__}: {exc}")

    report["ok"] = not errors
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check model files and policy I/O.")
    parser.add_argument("--root", type=Path, default=None, help="package root")
    args = parser.parse_args()
    report = verify_delivery(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
