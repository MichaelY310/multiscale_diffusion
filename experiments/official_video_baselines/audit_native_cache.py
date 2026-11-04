"""Bounded CPU native PF cache audit before its recursive-retry dataset loads."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from pyramid_smoke import PIN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--empty-prompt", type=Path, required=True)
    p.add_argument("--dit-config", type=Path, required=True)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU cache audit requires CUDA_VISIBLE_DEVICES empty")
    import torch
    cfg = json.loads(args.dit_config.read_text())
    dimensions = (cfg["joint_attention_dim"], cfg["pooled_projection_dim"])
    rows = [json.loads(line) for line in args.annotation.read_text().splitlines() if line.strip()]
    if not 4 <= len(rows) <= 128:
        raise ValueError("Use a dedicated small-data annotation with 4..128 rows")
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}
    if not args.synthetic and (provenance.get("source_commit") != PIN or provenance.get("synthetic") is not False):
        raise ValueError("Real cache audit requires native encoder provenance with fixed source commit")
    failures = []
    hashes = {str(args.annotation.resolve()): hashlib.sha256(args.annotation.read_bytes()).hexdigest()}
    def load(path):
        path = Path(path)
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return torch.load(path, map_location="cpu", weights_only=True)
    null = load(args.empty_prompt)
    sequence_length = null["prompt_embed"].shape[1]
    def text_valid(text):
        expected = {"prompt_embed": (1, sequence_length, dimensions[0]),
                    "prompt_attention_mask": (1, sequence_length),
                    "pooled_prompt_embed": (1, dimensions[1])}
        return all(key in text and tuple(text[key].shape) == shape and bool(torch.isfinite(text[key]).all())
                   for key, shape in expected.items())
    if not text_valid(null):
        failures.append({"row": "empty_prompt", "reason": "shape/dtype/finiteness"})
    for index, row in enumerate(rows):
        try:
            latent, text = load(row["latent"]), load(row["text_fea"])
            valid = latent.ndim == 5 and tuple(latent.shape[:2]) == (1, 16) and tuple(latent.shape[-2:]) == (48, 80)
            # Native Dataset deliberately clips longer raw VAE caches to
            # max_frames. Audit the original encoded tensor, not a recached
            # truncation with a different encoder provenance.
            valid = valid and 1 <= latent.shape[2] <= 128 and bool(torch.isfinite(latent).all())
            if not valid:
                failures.append({"row": index, "reason": "native 384p raw latent shape/finiteness", "shape": list(latent.shape)})
            if not text_valid(text):
                failures.append({"row": index, "reason": "text/cache empty-prompt shape/finiteness mismatch"})
        except Exception as exc:
            failures.append({"row": index, "reason": repr(exc)})
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    result = {"status": "failed" if failures else "passed", "synthetic": args.synthetic,
        "source_commit": PIN, "rows": len(rows), "files_sha256": hashes, "cache_fingerprint": fingerprint,
        "failures": failures, "cuda_initialized": torch.cuda.is_initialized(),
        "scope": "small cache loading/shapes/finiteness only; encoding and full-model update need their own tests",
        "encoder_provenance": provenance, "ready_for_formal_training": False}
    assert not result["cuda_initialized"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rows": len(rows), "failures": failures}))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
