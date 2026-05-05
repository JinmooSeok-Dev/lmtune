"""Static HTML dashboard (Phase W output G).

InferenceX-app 호환 JSON schema + Jinja2 templates → 정적 HTML.
사용:
    bench dashboard build --out b200/dashboards
    xdg-open b200/dashboards/index.html
"""

from lmtune.visualization.dashboard.build import build_dashboard, dump_inferencex_json

__all__ = ["build_dashboard", "dump_inferencex_json"]
