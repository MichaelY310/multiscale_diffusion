"""Small actual-checkpoint CUDA parity/backward gate before V3 training."""
import argparse
import json
from pathlib import Path

import torch

from csra_output_distillation import CSRAOutputStudent, prediction_loss
from sample_csra_adapter_preview import load_adapter
from train_rdd_csra_adapter import CSRAAdapterBank, initialize_adapter_weights, load_frozen_teacher


def main(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info()[0] < 12 * 2**30:
        raise RuntimeError("Need 12GiB headroom; never stop another process")
    manifest_path = Path(args.data_root) / "shard_0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    item = next(row for row in manifest["files"] if row["split"] == "validation")
    if Path(item["name"]).name != item["name"]:
        raise ValueError("Unexpected dataset path")
    data = torch.load(manifest_path.parent / item["name"], weights_only=True, map_location="cpu")
    teacher = load_frozen_teacher(Path(args.base_ckpt), device, "DiT-L/2", "rdd")
    v2, payload = load_adapter(Path(args.init_adapter), device)
    # Compare equal execution modes. A trainable versus frozen FP32 linear
    # can dispatch differently even under no_grad; that is not a V3 change.
    v3 = CSRAAdapterBank(v2.layers, payload["hidden_size"], payload["rank"], "teacher_condition_v3")
    v3 = v3.to(device).eval().requires_grad_(False)
    initialize_adapter_weights(v3, payload)
    x = torch.cat([data["x"][i, :2] for i in [0, 7]]).to(device)
    target = torch.cat([data["target"][i, :2] for i in [0, 7]]).to(device)
    t = torch.cat([data["t"][i, :2] for i in [0, 7]]).long().to(device)
    labels = data["labels"][:2].repeat(2).long().to(device)
    coarse = {l: data["coarse"][l][:2].repeat(2, 1, 1).to(device) for l in v2.layers}
    a, b = CSRAOutputStudent(teacher, v2), CSRAOutputStudent(teacher, v3)
    with torch.no_grad():
        reference = a(x, t, labels, coarse)
        prediction = b(x, t, labels, coarse)
        reference_again = a(x, t, labels, coarse)
    maximum = float((prediction-reference).abs().max())
    if maximum != 0:
        records = {}
        handles = []
        def record(arm, layer):
            def hook(module, inputs, output):
                records[(arm, layer)] = [item.detach().clone() for item in inputs] + [output.detach().clone()]
            return hook
        for arm, bank in [("v2", v2), ("v3", v3)]:
            for layer in bank.layers:
                handles.append(bank.adapters[str(layer)].register_forward_hook(record(arm, layer)))
        with torch.no_grad():
            a(x, t, labels, coarse)
            b(x, t, labels, coarse)
        for handle in handles:
            handle.remove()
        from csra_runtime import predict_adapter
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            condition = teacher.t_embedder(t) + teacher.y_embedder(labels, False)
        details = {}
        for layer in v2.layers:
            first, second = records[("v2", layer)], records[("v3", layer)]
            with torch.no_grad():
                one = predict_adapter(v2.adapters[str(layer)], *first[:3], precision="fp32")
                two = predict_adapter(v3.adapters[str(layer)], *first[:3], precision="fp32", condition=condition)
            details[layer] = {"input_difference": float((first[1]-second[1]).abs().max()),
                              "direct_delta_difference": float((one-two).abs().max()),
                              "trace_delta_difference": float((first[-1]-second[-1]).abs().max()),
                              "dtype": str(first[-1].dtype)}
        module_records = {}
        handles = []
        for arm, bank in [("v2", v2), ("v3", v3)]:
            for name, module in bank.adapters["13"].named_modules():
                if not name:
                    continue
                def hook(module, inp, out, arm=arm, name=name):
                    if torch.is_tensor(out):
                        module_records[(arm, name)] = (inp[0].detach().clone(), out.detach().clone(), inp[0].stride())
                handles.append(module.register_forward_hook(hook))
        inputs = records[("v2", 13)][:3]
        with torch.no_grad():
            for bank in [v2, v3]:
                predict_adapter(bank.adapters["13"], *inputs, precision="fp32", condition=condition)
        for handle in handles:
            handle.remove()
        submodules = {}
        for name, _ in v2.adapters["13"].named_modules():
            if ("v2", name) in module_records:
                left, right = module_records[("v2", name)], module_records[("v3", name)]
                submodules[name] = {"input_diff": float((left[0]-right[0]).abs().max()),
                                   "output_diff": float((left[1]-right[1]).abs().max()),
                                   "input_strides": [left[2], right[2]]}
        with torch.no_grad():
            v3.eval().requires_grad_(False)
            frozen = predict_adapter(v3.adapters["13"], *inputs, precision="fp32", condition=condition)
            frozen_diff = float((frozen - records[("v2", 13)][-1]).abs().max())
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            strict_left = predict_adapter(v2.adapters["13"], *inputs, precision="fp32", condition=condition)
            strict_right = predict_adapter(v3.adapters["13"], *inputs, precision="fp32", condition=condition)
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        print(json.dumps({"submodule_diagnostic": submodules, "v3_frozen_diff": frozen_diff,
                          "tf32_disabled_delta_difference": float((strict_left-strict_right).abs().max())}), flush=True)
        print(json.dumps({"v2_repeat_max_abs": float((reference-reference_again).abs().max()),
                          "parameter_equality": all(torch.equal(p, v3.state_dict()[k]) for k,p in v2.state_dict().items()),
                          "layer_diagnostic": details}), flush=True)
        raise RuntimeError(f"V2->V3 conversion changed actual prediction: {maximum}")
    v2.train().requires_grad_(True)
    v3.train().requires_grad_(True)
    with torch.no_grad():
        train_reference = a(x, t, labels, coarse)
        train_prediction = b(x, t, labels, coarse)
    training_maximum = float((train_prediction - train_reference).abs().max())
    if training_maximum != 0:
        raise RuntimeError(f"Trainable V2/V3 conversion changed prediction: {training_maximum}")
    loss, _ = prediction_loss(b(x, t, labels, coarse), target)
    loss.backward()
    gradients = {l: float(v3.adapters[str(l)].condition_mlp[-1].weight.grad.abs().sum()) for l in v3.layers}
    if any(not value > 0 for value in gradients.values()):
        raise RuntimeError(f"Condition gradient missing: {gradients}")
    if any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("Backbone was not frozen")
    print(json.dumps({"conversion_forward_max_abs": maximum,
                      "trainable_conversion_max_abs": training_maximum,
                      "v2_trainable_vs_frozen_max_abs": float((train_reference-reference).abs().max()),
                      "timesteps": t.tolist(),
                      "condition_grad_l1": gradients, "loss": float(loss.detach()),
                      "backbone_gradients": False,
                      "max_memory_gib": torch.cuda.max_memory_allocated()/2**30}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--data-root", required=True)
    main(parser.parse_args())
