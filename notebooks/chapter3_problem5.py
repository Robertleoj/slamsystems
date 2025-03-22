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
# %load_ext autoreload
# %autoreload 2
from project.utils.spatial.symbolic import hat, get_rotation_mat_constraints
import sympy as sp

# %%
sp.init_printing()

# %%
N = 3

# %%
r_elements = sp.symbols("r:3:3")
r_elements

# %%
R = sp.Matrix(3, 3, r_elements)
R

# %%
p_symbols = sp.symbols("p:3")
p_symbols

# %%
p_vec = sp.Matrix([p_symbols]).reshape(3, 1)
p_vec

# %%
p_hat = hat(p_vec)
p_hat

# %%
left_side = sp.simplify(R @ p_hat @ R.T)
left_side

# %%
right_side = sp.simplify(hat(R @ p_vec))
right_side
