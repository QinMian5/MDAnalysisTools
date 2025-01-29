# Author: Mian Qin
# Date Created: 2025/1/20
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

from utils import calculate_triangle_area
from physical_quantity import *
from utils import convert_unit


def reweight_free_energy(x: np.ndarray, free_energy: np.ndarray, T_src: float, T_dst: float, **kwargs):
    new_free_energy = free_energy.copy()

    # chemical potential
    delta_delta_mu = delta_mu_tip4p_ice(T_dst) - delta_mu_tip4p_ice(T_src)  # J/mol
    delta_delta_mu = delta_delta_mu / 1000  # to kJ/mol
    # delta_delta_mu = convert_unit(delta_delta_mu/1000, "kJ/mol", "kT")
    new_free_energy += delta_delta_mu * x

    # surface tension
    A_IW = kwargs.get("A_IW", 0)
    A_IS = kwargs.get("A_IS", 0)
    delta_gamma_IW = gamma_IL_tip4p_ice(T_dst) - gamma_IL_tip4p_ice(T_src)  # J/m^2
    delta_E = delta_gamma_IW * (A_IW - A_IS * 0.568148)  # J
    delta_E = delta_E / 1000 * c.N_A  # to kJ/mol
    new_free_energy += delta_E
    return new_free_energy


def main():
    x = np.arange(0, 1001, 100, dtype=float)
    free_energy = np.arange(0, 401, 40, dtype=float)
    T_src = 300
    T_dst = 250
    reweighted_free_energy = reweight_free_energy(x, free_energy, T_src, T_dst)
    print(free_energy)
    print(reweighted_free_energy)



if __name__ == "__main__":
    main()
