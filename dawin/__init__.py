from .core import DAWINLinear, compute_dawin_laplacian_loss
from .models import DAWINMLP, DAWINTransformer, DAWINGNN

try:
    from .triton_scan import TritonDAWINLinear, triton_dawin_scan
except ImportError:
    TritonDAWINLinear = None
    triton_dawin_scan = None

__version__ = "1.0.2"
__all__ = [
    "DAWINLinear",
    "TritonDAWINLinear",
    "triton_dawin_scan",
    "compute_dawin_laplacian_loss",
    "DAWINMLP",
    "DAWINTransformer",
    "DAWINGNN",
]

