"""Native randn_like and serialized streams must couple strided video tensors."""
import pytest
import torch
from umdiff.randomness import RandomStream


@pytest.mark.parametrize('layout',['contiguous','video_transposed','channels_last','sliced'])
def test_private_normal_matches_native_layout_and_rng(layout):
    x=torch.empty(2,4,3,8,12)
    if layout=='video_transposed':x=x.permute(0,2,1,3,4).contiguous().permute(0,2,1,3,4)
    elif layout=='channels_last':x=x.contiguous(memory_format=torch.channels_last_3d)
    elif layout=='sliced':x=x[:,:,:,:,::2]
    torch.manual_seed(730)
    expected=torch.randn_like(x);expected_state=torch.get_rng_state().clone()
    stream=RandomStream(730);actual=stream.normal(x)
    assert torch.equal(actual,expected)
    assert actual.stride()==expected.stride()
    assert torch.equal(stream.generator.get_state(),expected_state)
    assert torch.equal(torch.get_rng_state(),expected_state)
