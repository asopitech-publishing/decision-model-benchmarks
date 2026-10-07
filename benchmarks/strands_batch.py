"""Cross-request full-prompt batches for one Strands Decider MLX engine.

Question rendering, truncation and pointer positions use the installed Strands
implementation. The heavy MLX torso receives rows from independent requests
in one forward pass; no model replicas or HTTP concurrency are involved.
"""

from __future__ import annotations

import math


def predict_many(engine, packets: list[dict], *, max_rows: int) -> tuple[list[dict], int]:
    if not packets or max_rows < 1:
        raise ValueError("packets and max_rows must be nonempty/positive")

    import torch
    from strands_decider.infer import _to_answer
    from strands_decider.prompting import render_question, render_state
    from strands_decider.schema import SystemOneRequest

    responses = []
    rows = []
    with engine._lock:
        for request_index, packet in enumerate(packets):
            request = SystemOneRequest.model_validate(packet)
            names = list(request.questions)
            if not names:
                raise ValueError("each request needs at least one question")
            rendered = [render_question(request.questions[name]) for name in names]
            for question in rendered:
                if engine.model.config.head_type != "pointer" and question.n_slots > engine.model.config.num_slots:
                    raise ValueError("question has more options than model slots")
            state, questions = engine._fit(
                render_state(request.state), [question.text for question in rendered]
            )
            option_positions = (engine._option_idx(rendered, len(state)).tolist()
                                if engine.model.config.head_type == "pointer" else [None] * len(names))
            responses.append({"model": engine.cfg.model_name, "answers": {},
                              "usage": {"input_tokens": sum(len(state) + len(q) for q in questions),
                                        "output_tokens": len(names)}})
            rows.extend((request_index, name, question, state + token_ids, positions)
                        for name, question, token_ids, positions in
                        zip(names, rendered, questions, option_positions, strict=True))

        forward_calls = 0
        for start in range(0, len(rows), max_rows):
            chunk = rows[start : start + max_rows]
            sequences = [sequence for _, _, _, sequence, _ in chunk]
            if engine.model.config.head_type == "pointer":
                width = max(len(positions) for *_, positions in chunk)
                opt_idx = torch.tensor(
                    [positions + [-1] * (width - len(positions)) for *_, positions in chunk],
                    dtype=torch.long,
                )
            else:
                opt_idx = None
            hidden = engine._hidden(sequences)
            probabilities = engine._probs(
                hidden, [len(sequence) - 1 for sequence in sequences], opt_idx,
                [question.n_slots for _, _, question, _, _ in chunk],
                [question.kind for _, _, question, _, _ in chunk],
            )
            forward_calls += 1
            for row_index, (request_index, name, question, _, _) in enumerate(chunk):
                values = probabilities[row_index, :question.n_slots].tolist()
                answer = _to_answer(
                    question, values,
                    ordinal_smoothing=engine.model.config.ordinal_smoothing,
                )
                responses[request_index]["answers"][name] = answer.model_dump()

    assert forward_calls == math.ceil(len(rows) / max_rows)
    return responses, forward_calls
