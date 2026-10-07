"""Cross-request batches for one resident laya-mlx Agent (Apache-2.0).

Each question from every request becomes one row of the same MLX forward pass.
This uses the installed laya-mlx Agent's public preparation and forward methods;
it does not create more model instances or change HTTP admission behavior.
"""

from __future__ import annotations

import math


def predict_many(agent, packets: list[dict], *, max_rows: int) -> tuple[list[dict], int]:
    """Return System One responses and the number of actual model forward calls."""
    if not packets or max_rows < 1:
        raise ValueError("packets and max_rows must be nonempty/positive")

    import numpy as np
    from laya_mlx.agent import collate_items
    from laya_mlx.common import confidence_from_probs, temp_bucket

    responses = []
    rows = []
    for request_index, packet in enumerate(packets):
        questions = packet["questions"]
        items, internal = agent.prepare(packet["state"], questions)
        if not items:
            raise ValueError("each request needs at least one question")
        responses.append({
            "model": "laya-rl-agent", "answers": {},
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        })
        rows.extend((request_index, qid, q, item)
                    for qid, q, item in zip(questions, internal, items, strict=True))

    forward_calls = 0
    for start in range(0, len(rows), max_rows):
        chunk = rows[start : start + max_rows]
        batch = collate_items(
            [item for _, _, _, item in chunk], agent.tok.pad_token_id,
            pad_to_multiple=agent.pad_to_multiple,
            max_length=agent.cfg.get("max_len", 512),
        )
        logits, act = agent.forward(batch)
        forward_calls += 1
        logits, act = np.asarray(logits), np.asarray(act)
        if not np.isfinite(logits).all() or not np.isfinite(act).all():
            raise FloatingPointError("Non-finite model outputs")
        act = np.exp(act - act.max(axis=-1, keepdims=True))
        act /= act.sum(axis=-1, keepdims=True)
        for row_index, (request_index, qid, q, item) in enumerate(chunk):
            k, qt = len(item["markers"]), item["qtype"]
            scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            z = logits[row_index, :k] / scale
            probs = np.exp(z - z.max())
            probs /= probs.sum()
            answer = {
                "type": q["t"],
                "confidence": round(confidence_from_probs(probs, k), 4),
                "action": {"act_probability": round(float(act[row_index, 0]), 4)},
            }
            if q["t"] == "choice":
                labels = list(q["crit"])
                answer.update(
                    choice=labels[int(probs.argmax())],
                    probabilities={label: round(float(value), 4)
                                   for label, value in zip(labels, probs)},
                )
            elif q["t"] == "score":
                answer.update(
                    score=round(float((np.arange(k) * probs).sum()), 4),
                    legend={str(i): value for i, value in enumerate(q["crit"])},
                    probabilities={str(i): round(float(value), 4)
                                   for i, value in enumerate(probs)},
                )
            else:
                answer.update(
                    noul=round(float(probs[1]), 4),
                    confidence=round(max(float(probs[1]), 1.0 - float(probs[1])), 4),
                )
            responses[request_index]["answers"][qid] = answer

    assert forward_calls == math.ceil(len(rows) / max_rows)
    return responses, forward_calls
