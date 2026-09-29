"""Batch-shaped JSON transport with row-isolated validation and retry."""

from __future__ import annotations

import json


def left_pad_rows(rows, pad_token_id: int, torch_module, device: str):
    if not isinstance(rows, list) or not rows or any(not isinstance(row, list) or not row for row in rows):
        raise ValueError("token rows必须是非空二维数组")
    width = max(len(row) for row in rows)
    input_ids = torch_module.full((len(rows), width), pad_token_id, dtype=torch_module.long, device=device)
    attention_mask = torch_module.zeros((len(rows), width), dtype=torch_module.long, device=device)
    lengths = []
    for index, row in enumerate(rows):
        length = len(row)
        lengths.append(length)
        input_ids[index, width - length :] = torch_module.tensor(row, dtype=torch_module.long, device=device)
        attention_mask[index, width - length :] = 1
    useful = sum(lengths)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_lengths": lengths,
        "padded_input_tokens": len(rows) * width - useful,
        "padded_width": width,
    }


def call_json_batch(llm, requests: list[dict], max_output: int = 768, validators=None):
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests不能为空")
    validators = [None] * len(requests) if validators is None else validators
    if not isinstance(validators, list) or len(validators) != len(requests):
        raise ValueError("requests与validators数量必须一致")
    for request in requests:
        if (
            not isinstance(request, dict)
            or not isinstance(request.get("request_id"), str)
            or not isinstance(request.get("messages"), list)
        ):
            raise ValueError("batch request字段无效")
    values = [None] * len(requests)
    row_attempts: list[list[dict]] = [[] for _ in requests]
    active = list(range(len(requests)))
    current_messages = {index: list(requests[index]["messages"]) for index in active}
    batch_calls = []
    for attempt_number in (1, 2):
        if not active:
            break
        messages = [current_messages[index] for index in active]
        try:
            outputs = llm._generate_batch(messages, max_new_tokens=max_output)
            if not isinstance(outputs, list) or len(outputs) != len(active):
                raise RuntimeError("批量生成输出数量与输入不一致")
            timing = dict(getattr(llm, "last_batch_generation", {}))
            truncated_positions = set(timing.get("truncated_indices", []))
            batch_calls.append({"attempt": attempt_number, "request_ids": [requests[index]["request_id"] for index in active], "timing": timing, "status": "ok"})
        except (ValueError, RuntimeError, OSError) as exc:
            error = str(exc)
            batch_calls.append({"attempt": attempt_number, "request_ids": [requests[index]["request_id"] for index in active], "timing": dict(getattr(llm, "last_batch_generation", {})), "status": "error", "error": error, "error_type": type(exc).__name__})
            for index in active:
                row_attempts[index].append({"attempt": attempt_number, "status": "error", "raw_output": "", "error": error, "error_type": type(exc).__name__})
            active = []
            break
        failed = []
        for position, (index, raw) in enumerate(zip(active, outputs)):
            try:
                if position in truncated_positions:
                    raise ValueError("生成达到输出上限且未结束")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("输出必须是JSON对象")
                validator = validators[index]
                if validator is not None:
                    validator(value)
                values[index] = value
                row_attempts[index].append({"attempt": attempt_number, "status": "ok", "raw_output": raw})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                error = str(exc)
                row_attempts[index].append({"attempt": attempt_number, "status": "error", "raw_output": getattr(exc, "raw_output", raw), "error": error, "error_type": type(exc).__name__})
                if attempt_number == 1:
                    current_messages[index] = list(requests[index]["messages"]) + [
                        {"role": "user", "content": "上次回复格式不合格：" + error + "。重新只返回符合要求的一个完整JSON对象，不加说明。"}
                    ]
                    failed.append(index)
        active = failed
    rows = []
    for index, request in enumerate(requests):
        attempts = row_attempts[index]
        row = {
            "request_id": request["request_id"],
            "status": "ok" if values[index] is not None else "error",
            "attempts": len(attempts),
            "attempt_records": attempts,
        }
        if values[index] is None:
            row["error"] = attempts[-1]["error"] if attempts else "未执行"
        rows.append(row)
    succeeded = sum(value is not None for value in values)
    status = "ok" if succeeded == len(values) else "error" if succeeded == 0 else "partial"
    return values, {
        "status": status,
        "rows": rows,
        "batch_calls": batch_calls,
        "retry_count": sum(len(attempts) > 1 for attempts in row_attempts),
    }
