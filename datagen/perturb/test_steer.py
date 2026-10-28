"""E67's two pieces of model surgery, on a tiny random Qwen3.5 on CPU: the grouped decode attention changes no sampled
token, and the push moves only the steered layer's prefill states.

Run:  python datagen/perturb/test_steer.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AttentionInterface, Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from steer import Steer, decoder_layers, grouped_decode_sdpa  # noqa: E402


def tiny():
    config = Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                               num_attention_heads=8, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                               linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                               layer_types=["linear_attention"] * 3 + ["full_attention"], attn_implementation="sdpa")
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).eval()


def sample(model, ids):
    torch.manual_seed(1)
    with torch.no_grad():
        return model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True, top_k=40, top_p=0.95,
                              max_new_tokens=40, pad_token_id=0, eos_token_id=None)


def test_grouped_decode_is_exact():
    model, ids = tiny(), torch.randint(0, 300, (3, 12))
    original = AttentionInterface._global_mapping["sdpa"]
    try:
        before = sample(model, ids)
        grouped_decode_sdpa()
        assert torch.equal(before, sample(model, ids))
    finally:
        AttentionInterface.register("sdpa", original)


def test_push_moves_only_the_steered_prefill():
    model, ids = tiny(), torch.randint(0, 300, (2, 12))
    steer = Steer()
    decoder_layers(model, 4)[1].register_forward_hook(steer)  # its output is hidden-state index 2
    delta = torch.zeros(2, 64)
    delta[1, 0] = 5.0

    def states(tokens, push):
        steer.delta = push
        with torch.no_grad():
            return model(input_ids=tokens, output_hidden_states=True).hidden_states

    pushed, plain = states(ids, delta), states(ids, None)
    assert torch.allclose(pushed[1], plain[1]) and torch.allclose(pushed[2][0], plain[2][0])
    assert torch.allclose(pushed[2][1] - plain[2][1], delta[1].expand(12, 64), atol=1e-5)
    assert torch.allclose(states(ids[:, :1], delta)[2], states(ids[:, :1], None)[2])  # a one-token (decode) pass


if __name__ == "__main__":
    test_grouped_decode_is_exact()
    test_push_moves_only_the_steered_prefill()
    print("ok")
