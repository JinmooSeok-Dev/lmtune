from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_PATTERNS = [
    # vLLM 0.13+ 예: "Received request cmpl-xxx: prompt=..., sampling_params=..., prompt_token_ids=..."
    re.compile(r"Received request (?P<req_id>[\w\-]+):.*?prompt_token_ids=\[(?P<tokens>[^\]]*)\]"),
    # 완료 로그: "Finished request cmpl-xxx in N.NNNs"
    re.compile(r"Finished request (?P<req_id>[\w\-]+) in (?P<elapsed>[\d.]+)s"),
]


def parse_request_log(path: str | Path) -> list[dict]:
    events: list[dict] = []
    text = Path(path).read_text(errors="ignore").splitlines()
    for line in text:
        for pat in _PATTERNS:
            m = pat.search(line)
            if m:
                d = m.groupdict()
                if "tokens" in d and d["tokens"]:
                    d["prompt_tokens"] = d["tokens"].count(",") + 1 if d["tokens"].strip() else 0
                    d.pop("tokens", None)
                events.append({"raw": line, **d})
                break
    return events


def summarize(events: Iterable[dict]) -> dict:
    req_ids = {e["req_id"] for e in events if "req_id" in e}
    prompt_tokens = [int(e["prompt_tokens"]) for e in events if "prompt_tokens" in e]
    elapsed = [float(e["elapsed"]) for e in events if "elapsed" in e]
    return {
        "unique_requests": len(req_ids),
        "prompt_tokens_mean": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
        "elapsed_mean_sec": (sum(elapsed) / len(elapsed)) if elapsed else None,
    }
