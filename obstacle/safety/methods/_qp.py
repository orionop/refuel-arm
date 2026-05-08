"""Thin QP wrapper used by NEO + HOCBF.

We try `qpsolvers` first (unified API over OSQP/quadprog/CVXOPT). If it isn't
installed we fall back to a `scipy.optimize.minimize` SLSQP solve, which is
slower but always available. This keeps the safety methods runnable on a fresh
laptop while letting the production deployment pin OSQP for speed.

Problem solved:

    min_x  ½ xᵀ H x + gᵀ x
    s.t.   A x ≤ b              (optional)
           lb ≤ x ≤ ub          (box)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from qpsolvers import solve_qp as _solve_qp  # type: ignore
    _HAVE_QPSOLVERS = True
except Exception:                                 # pragma: no cover
    _HAVE_QPSOLVERS = False


def solve_qp_box(
    H: np.ndarray,
    g: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
) -> Tuple[Optional[np.ndarray], str]:
    """Return (x, status). status is "ok", solver name, or "fallback"."""
    H = 0.5 * (H + H.T)  # symmetrise

    if _HAVE_QPSOLVERS:
        try:
            x = _solve_qp(H, g, G=A, h=b, lb=lb, ub=ub, solver="osqp")
            if x is not None:
                return np.asarray(x, dtype=float), "osqp"
        except Exception:
            pass
        try:
            x = _solve_qp(H, g, G=A, h=b, lb=lb, ub=ub, solver="quadprog")
            if x is not None:
                return np.asarray(x, dtype=float), "quadprog"
        except Exception:
            pass

    # ── fallback: SLSQP via scipy. Slower, but no extra deps. ─────────────
    from scipy.optimize import minimize  # local import keeps import cost lazy

    n = H.shape[0]
    x0 = np.clip(-np.linalg.solve(H + 1e-8 * np.eye(n), g), lb, ub)

    def obj(x):
        return 0.5 * x @ H @ x + g @ x

    def grad(x):
        return H @ x + g

    constraints = []
    if A is not None and b is not None:
        for ai, bi in zip(A, b):
            constraints.append({"type": "ineq", "fun": (lambda x, a=ai, c=bi: c - a @ x),
                                "jac": (lambda x, a=ai: -a)})

    res = minimize(
        obj, x0, jac=grad, method="SLSQP",
        bounds=list(zip(lb, ub)), constraints=constraints,
        options={"maxiter": 80, "ftol": 1e-6},
    )
    if not res.success:
        return None, f"slsqp:{res.message}"
    return np.asarray(res.x, dtype=float), "slsqp"
