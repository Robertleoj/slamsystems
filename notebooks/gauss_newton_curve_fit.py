# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import symforce
if 'eps_set' not in globals():
    symforce.set_epsilon_to_symbol()
    eps_set = True

from symforce.notebook_util import print_expression_tree, set_notebook_defaults
set_notebook_defaults()

import numpy as np
import einops
import symforce.symbolic as sf
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 10, 50)
y = np.sin(x)
plt.plot(x, y)

# %%
order = 6


# %%
class CoeffMatrix(sf.Matrix):
    SHAPE = (order + 1, 1)


# %%
coeffs = CoeffMatrix.symbolic( f"C")
coeffs

# %%
x_symb = sf.Symbol('x')
y_symb = sf.Symbol('y')

poly = 0.0

for i in range(order + 1):
    if i == 0:
        poly = poly + coeffs[i]
        continue

    poly = poly + sf.Pow(x_symb, float(i)) * coeffs[i]

poly = sf.V1(poly)
print(type(poly))
poly

# %%
residual = sf.V1(y_symb) - poly

# %%
J_coeffs = residual.jacobian(coeffs)
J_coeffs


# %%
def eval_poly(x_pow_mat: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return einops.einsum(x_pow_mat, coeffs, "d N, d -> N")


# %%
curr_coeffs = np.zeros(order + 1)

N = x.shape[0]

x_pow_mat = x.reshape(1, -1) ** np.array(range(order + 1)).reshape(-1, 1)
deltas = []

for i in range(10):

    y_hat = eval_poly(x_pow_mat, curr_coeffs)
    y_hat = einops.einsum(x_pow_mat, curr_coeffs, "d N, d -> N")
    residuals = y - y_hat

    H = np.zeros((order + 1, order + 1))
    g = np.zeros(order + 1)

    for i in range(N):
        jacobian = np.array(J_coeffs.subs({x_symb: x[i]}), dtype=float)

        g += - jacobian * residuals[i]

        H += jacobian[:, None] * jacobian[None, :]

    delta_coeffs = np.linalg.solve(H, g)
    deltas.append(delta_coeffs)


    curr_coeffs = curr_coeffs + delta_coeffs
        

# %%
deltas = np.array(deltas)
norms = np.linalg.norm(deltas, axis=1)
plt.plot(norms)

# %%
curr_coeffs

# %%
y_hat = eval_poly(x_pow_mat, curr_coeffs)
plt.plot(x, y_hat, label="pred")
plt.plot(x, y, label='real')
plt.legend()
