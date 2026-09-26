"""Tokenizer loading that refuses to run on a tokenizer that does not round-trip.

Under the pinned runtime (transformers 5.13.0), `AutoTokenizer` resolves
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` to a legacy `LlamaTokenizer` conversion that drops
every space and newline: "Every morning Aya" encodes, and decodes, as "EverymorningAya".
The starter kit's uncertainty baseline works around this by loading
`PreTrainedTokenizerFast` directly; its trained-probe baseline does not. E1 was extracted
with the broken tokenizer (PROGRESS.md Finding 14).

Every script that tokenizes model input loads through here, so a silent tokenizer failure
becomes a hard error instead of corrupted features.
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

ROUND_TRIP_SAMPLE = "Every morning Aya goes for a $9$-kilometer-long walk.\nFind the answer \\boxed{204}."


def load_checked_tokenizer(model_id: str, allow_lossy: bool = False, **kwargs) -> tuple[PreTrainedTokenizerBase, dict]:
    """Load the fast byte-level tokenizer (falling back to AutoTokenizer), verify the encode/decode
    round trip, and return provenance for metadata.json."""
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, **kwargs)
    except (OSError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    decoded = tokenizer.decode(tokenizer(ROUND_TRIP_SAMPLE, add_special_tokens=False)["input_ids"])
    ok = decoded == ROUND_TRIP_SAMPLE
    if not ok and not allow_lossy:
        raise SystemExit(
            f"tokenizer {type(tokenizer).__name__} for {model_id} does not round-trip: "
            f"{ROUND_TRIP_SAMPLE!r} -> {decoded!r}"
        )
    return tokenizer, {"tokenizer_class": type(tokenizer).__name__, "tokenizer_round_trip": ok}
