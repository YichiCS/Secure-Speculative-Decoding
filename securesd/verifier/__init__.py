from .common import verify_lossy_sd, verify_sd
from .bild import verify_bild
from .fly import verify_fly
from .fsd import verify_fsd
from .mars import verify_mars
from .sc import verify_sc

from .common import resolve_all_greedy
from .fusion import apply_eta_fusion
from .registry import get_impl
from .types import VerifierMethodOutput, VerifierMethodRequest

__all__ = [
    "run_verifier_method",
    "VerifierMethodRequest",
    "VerifierMethodOutput",
]


def run_verifier_method(request: VerifierMethodRequest) -> VerifierMethodOutput:
    impl = get_impl(request.method)

    if request.eta is None:
        return impl(request)

    if request.eta_bypass == 0.0:
        return verify_sd(request)
    if request.eta_bypass == 1.0 or request.method == "sd":
        return impl(request)

    all_greedy = resolve_all_greedy(request, None)
    native = impl(request, probs_only=True, all_greedy=all_greedy)
    return apply_eta_fusion(request, native, all_greedy=all_greedy)
