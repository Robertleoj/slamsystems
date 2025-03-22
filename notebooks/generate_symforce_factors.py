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
symforce.set_epsilon_to_symbol()
from project.utils.symforce_utils.factors import generate_point_reprojection_factor, generate_depth_factor


generate_point_reprojection_factor()
generate_depth_factor()
