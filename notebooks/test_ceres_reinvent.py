# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: project-ePUsKrUH-py3.11
#     language: python
#     name: python3
# ---

# %%
from project.foundation.reinventing import fit_poly_ceres, fit_poly_g2o
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 10.0, 100)
y = np.sin(x)

# %%
plt.plot(x, y)

# %%

# %%
order_to_coeffs = {}
order_to_g2o_coeffs = {}
for order in range(1, 6):
    order_to_coeffs[order] = fit_poly_ceres(x.tolist(), y.tolist(), order)
    order_to_g2o_coeffs[order] = fit_poly_g2o(x.tolist(), y.tolist(), order)

# %%
order_to_coeffs


# %%
@dataclass(frozen=True)
class Polynomial:
    coeffs: list[float]

    def eval(self, x: np.ndarray):
        x = x.flatten()
        assert len(x.shape) == 1

        y = np.zeros_like(x)
        for i, coeff in enumerate(self.coeffs):
            if i == 0:
                y += coeff
                continue
            y += coeff * (x ** i)

        return y
            


# %%
ceres_polynomials = {order: Polynomial(coeffs) for order, coeffs in order_to_coeffs.items()}
for order, poly in ceres_polynomials.items():
    y_hat = poly.eval(x)
    plt.plot(x, y_hat, label=f"order_{order}")

plt.legend()


# %%
g2o_polynomials = {order: Polynomial(coeffs) for order, coeffs in order_to_g2o_coeffs.items()}
for order, poly in g2o_polynomials.items():
    y_hat = poly.eval(x)
    plt.plot(x, y_hat, label=f"order_{order}")

plt.legend()


