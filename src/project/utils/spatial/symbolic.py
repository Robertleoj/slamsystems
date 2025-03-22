from typing import cast

import sympy as sp


def hat(mat: sp.Matrix):
    assert mat.shape == (3, 1)

    p1: sp.Symbol = cast(sp.Symbol, mat[0, 0])
    p2 = cast(sp.Symbol, mat[1, 0])
    p3 = cast(sp.Symbol, mat[2, 0])

    return sp.Matrix([[0, -p3, p2], [p3, 0, -p1], [-p2, p1, 0]])
