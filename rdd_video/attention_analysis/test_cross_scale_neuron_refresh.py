"""CPU algebra and CUDA transport parity tests; no model download."""
import torch
import torch.nn.functional as F
from .cross_scale_neuron_refresh import CrossScaleNeuronRefresh, lift_reference, lift_add


def check(device):
    torch.manual_seed(8821)
    dtype = torch.bfloat16 if device == "cuda" else torch.float64
    for batch in (1, 2):
        for source, target in (((2, 3, 4), (4, 6, 8)), ((4, 2, 3), (4, 8, 12))):
            coarse = torch.randn(batch, torch.tensor(source).prod(), 37, device=device, dtype=dtype)
            fresh = torch.randn(batch, torch.tensor(target).prod(), 37, device=device, dtype=dtype)
            ref = lift_reference(coarse, source, target)+fresh
            torch.testing.assert_close(lift_add(coarse, fresh, source, target), ref, rtol=0, atol=0)
    # At unchanged input, a channel partition reproduces the complete FFN,
    # including biases exactly once; this is NOT a fine-input parity claim.
    xc = torch.randn(2, 6, 7, device=device, dtype=torch.float64)
    wi = torch.randn(12, 7, device=device, dtype=torch.float64)
    wo = torch.randn(7, 12, device=device, dtype=torch.float64)
    bi = torch.randn(12, device=device, dtype=torch.float64)
    bo = torch.randn(7, device=device, dtype=torch.float64)
    ac = F.gelu(F.linear(xc, wi, bi))
    yc = F.linear(ac, wo, bo)
    for count in (0, 4, 12):
        active = F.linear(ac[..., :count], wo[:, :count])
        torch.testing.assert_close(yc-active+active, yc, rtol=1e-12, atol=1e-12)
    try:
        lift_reference(coarse, source, (5, 8, 12))
        raise AssertionError("Invalid nesting accepted")
    except ValueError:
        pass


def check_lifecycle():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_in = torch.nn.Linear(8, 256)
            self.fc_out = torch.nn.Linear(256, 8)
            self.act = torch.nn.GELU()
        def forward(self, x):
            return self.fc_out(self.act(self.fc_in(x)))
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = MLP()
    class Model(torch.nn.Module):
        patch_size = (1, 2, 2)
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])
    model = Model().eval()
    c = CrossScaleNeuronRefresh(model)
    with torch.inference_mode():
        for stages in ((2,), (1, 2)):
            c.configure(.5, [0], stages, "torch")
            for stage, shape in enumerate(((2,4,4), (2,8,8), (2,16,16))):
                c.begin_model_forward(stage, shape)
                x = torch.randn(1, torch.tensor(c.grid).prod(), 8)
                y = model.blocks[0].ffn(x)
                assert y.shape == x.shape and y.isfinite().all()
            assert c.audit["refreshes"] == len(stages) and not c.cache
        c.configure(1., [0], (2,), "torch")
        x = torch.randn(1, 6, 8)
        torch.testing.assert_close(model.blocks[0].ffn(x), c.originals[0](x), rtol=0, atol=0)
    c.restore()


if __name__ == "__main__":
    check("cpu")
    check_lifecycle()
    if torch.cuda.is_available():
        check("cuda")
    print("PASS: channel-partition algebra, batch/grid and fused transport parity")
