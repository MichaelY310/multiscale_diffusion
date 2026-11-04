"""Dependency-free LoRA tests."""

import torch
from torch import nn

from lora_utils import inject_lora, lora_state_dict, merge_lora_


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.q = nn.Linear(8, 8)
        block.self_attn.k = nn.Linear(8, 8)
        block.self_attn.v = nn.Linear(8, 8)
        block.self_attn.o = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([block])

    def forward(self, x):
        attn = self.blocks[0].self_attn
        return attn.o(attn.q(x) + attn.k(x) + attn.v(x))


def main() -> None:
    torch.manual_seed(0)
    model = TinyWan()
    report = inject_lora(model, rank=2, alpha=2)
    assert len(report.names) == 4
    assert report.trainable_parameters > 0
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    x = torch.randn(3, 8)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_b" in name:
                parameter.normal_(std=0.01)
    expected = model(x)
    state = lora_state_dict(model)
    assert state and all(tensor.device.type == "cpu" for tensor in state.values())
    assert merge_lora_(model) == 4
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    print("LoRA CPU tests passed")


if __name__ == "__main__":
    main()
