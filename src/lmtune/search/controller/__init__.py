"""Controller — Layer 4 plug-in seam for "next params" decision.

Study 가 (persistence + breaker + profile_binder) 를 담당하고, "next params"
결정만 Controller 에 위임. Optuna 락인 해제 — 외부 LLM/agent API, 자체 RL,
mock 구현 등 어떤 것도 plug-in 가능.

3 reference 구현 + 1 mock:
- OptunaController : 현재 sampler 8종 (TPE/CMA-ES/NSGA-II/...) wrap (기본값)
- RandomController : Optuna 의존성 0, pure Python (검증·baseline)
- HTTPController   : URL 에 POST, 외부 LLM/agent API 통합 base
- MockController   : scripted params 순차 반환 (테스트·minikube smoke)

contract — `docs/architecture.md` § Layer 4 Pluggability 참조.
"""

from lmtune.search.controller.base import Controller
from lmtune.search.controller.http_ctrl import HTTPController
from lmtune.search.controller.mock_ctrl import MockController
from lmtune.search.controller.random_ctrl import RandomController

# OptunaController 는 optuna 의존성이 있어 lazy — `[search]` extra 미설치 환경에서도
# Random/Mock/HTTP plug-in 은 동작해야 한다.
try:
    from lmtune.search.controller.optuna_ctrl import OptunaController  # noqa: F401
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

__all__ = [
    "Controller",
    "RandomController",
    "HTTPController",
    "MockController",
    "make_controller",
]
if _HAS_OPTUNA:
    __all__.append("OptunaController")


def make_controller(
    kind: str,
    space,
    *,
    strategy: str = "tpe",
    seed: int | None = None,
    context: dict | None = None,
    n_samples: int | None = None,
    direction: str = "maximize",
    directions: list[str] | None = None,
    study_name: str | None = None,
    pruner: str | None = None,
    url: str | None = None,
    scripted_params: list[dict] | None = None,
) -> Controller:
    """CLI 친화 factory. `--controller {optuna,random,http,mock}` 매핑."""
    kind = kind.lower()
    if kind in ("optuna", "default"):
        if not _HAS_OPTUNA:
            raise ImportError(
                "OptunaController requires optuna. install with `pip install lmtune[search]`"
            )
        return OptunaController.from_config(
            space, strategy=strategy, seed=seed, context=context,
            n_samples=n_samples, direction=direction, directions=directions,
            study_name=study_name, pruner=pruner,
        )
    if kind == "random":
        return RandomController(seed=seed)
    if kind == "http":
        if not url:
            raise ValueError("HTTPController requires --controller-url")
        return HTTPController(url=url, study_id=study_name or "anon")
    if kind == "mock":
        return MockController(scripted_params=scripted_params)
    raise ValueError(f"unknown controller kind: {kind!r}. Choose from optuna/random/http/mock")
