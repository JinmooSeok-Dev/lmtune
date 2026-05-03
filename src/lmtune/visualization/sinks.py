"""결과 출력 sinks — md / html / csv / parquet / jupyter."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

_SINKS: dict[str, Callable] = {}


def register_sink(name: str):
    def wrap(fn):
        _SINKS[name] = fn
        return fn
    return wrap


def list_sinks() -> list[str]:
    return sorted(_SINKS)


def write(sink: str, df: pd.DataFrame, out_path: str | Path, **opts):
    fn = _SINKS.get(sink)
    if fn is None:
        raise ValueError(f"unknown sink: {sink}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return fn(df, out_path, **opts)


@register_sink("csv")
def _sink_csv(df: pd.DataFrame, out: Path, **_):
    df.to_csv(out, index=False)
    return out


@register_sink("parquet")
def _sink_parquet(df: pd.DataFrame, out: Path, **_):
    df.to_parquet(out, index=False)
    return out


@register_sink("json")
def _sink_json(df: pd.DataFrame, out: Path, **_):
    df.to_json(out, orient="records", indent=2)
    return out


@register_sink("markdown")
def _sink_md(df: pd.DataFrame, out: Path, title: str = "", **_):
    text = f"# {title}\n\n{df.to_markdown(index=False)}\n" if title else df.to_markdown(index=False)
    out.write_text(text, encoding="utf-8")
    return out


@register_sink("html")
def _sink_html(df: pd.DataFrame, out: Path, title: str = "", **_):
    body = df.to_html(index=False, escape=True)
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title or 'bench report'}</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:2em auto}}
table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:4px 8px}}
</style></head><body>
<h1>{title or 'bench report'}</h1>
{body}
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    return out


@register_sink("jupyter")
def _sink_jupyter(df: pd.DataFrame, out: Path, title: str = "", **_):
    """간단한 ipynb 생성 (외부 nbformat 없이)."""
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {title or 'bench report'}"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [
             "import pandas as pd\n",
             f"df = pd.read_csv({json.dumps(str(out.with_suffix('.csv')))!s})\n",
             "df.describe()\n",
         ]},
    ]
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    return out
