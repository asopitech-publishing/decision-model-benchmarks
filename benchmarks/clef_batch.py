"""Cross-request text batches for one Clef-flash MLX model (Apache-2.0).

The Qwen torso runs once for several padded records. Clef's joint schema head
still processes each record's questions separately, matching its public API.
"""

from __future__ import annotations

import math


def predict_many(model, packets: list[dict], *, max_rows: int) -> tuple[list[dict], int]:
    if not packets or max_rows < 1:
        raise ValueError("packets and max_rows must be nonempty/positive")

    import mlx.core as mx
    from clef_mlx import encode_record, systemone_answer

    encoded = []
    for packet in packets:
        if packet.get("images") or packet.get("videos"):
            raise ValueError("this benchmark uses text-only records")
        record = {"model": "clef-flash-4bit", **packet}
        enc = encode_record(model.tokenizer, record, processor=model.processor,
                            max_length=16384, truncate=True)
        encoded.append((record, enc))

    responses = []
    forward_calls = 0
    for start in range(0, len(encoded), max_rows):
        chunk = encoded[start : start + max_rows]
        width = max(len(enc.input_ids) for _, enc in chunk)
        pad_id = model.tokenizer.pad_token_id or 0
        ids = mx.array([list(enc.input_ids) + [pad_id] * (width - len(enc.input_ids))
                        for _, enc in chunk], dtype=mx.int32)
        positions = mx.broadcast_to(mx.arange(width, dtype=mx.int32)[None, None, :],
                                    (3, len(chunk), width))
        hidden = model._text_model(ids, position_ids=positions)
        mx.eval(hidden)
        forward_calls += 1
        for row, (record, enc) in enumerate(chunk):
            actual_ids = mx.array(enc.input_ids, dtype=mx.int32)
            logits = model.head(hidden[row, :len(enc.input_ids)], actual_ids, enc, model._lexical)
            mx.eval(logits)
            answers = {
                question.question_id: systemone_answer(
                    record["questions"][question.question_id],
                    dict(zip(question.option_ids, mx.softmax(logit.astype(mx.float32)).tolist())),
                )
                for question, logit in zip(enc.questions, logits, strict=True)
            }
            responses.append({"model": record["model"], "answers": answers,
                              "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}})

    assert forward_calls == math.ceil(len(encoded) / max_rows)
    return responses, forward_calls
