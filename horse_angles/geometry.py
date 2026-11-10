"""Pure joint geometry."""
import math
from collections.abc import Sequence


def compute_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float | None:
    """Return the interior angle at b, or None for degenerate vectors."""
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    lengths = math.hypot(*u), math.hypot(*v)
    if min(lengths) < 1e-6:
        return None
    cosine = sum(x * y for x, y in zip(u, v)) / (lengths[0] * lengths[1])
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
