# Author: Mian Qin
# Date Created: 2025/1/20
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
from scipy.optimize import newton
from scipy.interpolate import interp1d
import uncertainties.unumpy as unp

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
    x_A = kwargs.get("x_A", None)
    A_IW = kwargs.get("A_IW", None)
    A_IS = kwargs.get("A_IS", None)
    if x_A is not None:
        delta_E = 0
        delta_gamma_IW = gamma_IL_tip4p_ice(T_dst) - gamma_IL_tip4p_ice(T_src)  # J/m^2
        if A_IW is not None:
            f_IW = interp1d(x_A, A_IW)
            A_IW_interp = f_IW(x)
            delta_E += delta_gamma_IW * A_IW_interp  # J
        if A_IS is not None:
            f_IS = interp1d(x_A, A_IS)
            A_IS_interp = f_IS(x)
            delta_E += - delta_gamma_IW * A_IS_interp * 0.568148
        delta_E = delta_E / 1000 * c.N_A  # to kJ/mol
        new_free_energy += delta_E
    return new_free_energy


def get_delta_T_star(x: np.ndarray, free_energy: np.ndarray, T_sim: float, **kwargs):
    free_energy = unp.nominal_values(free_energy)
    G_barr_threshold = 5  # kT
    T_m = T_m_tip4p_ice.value
    G_barr_initial = convert_unit(free_energy.max(), T=T_sim)
    if G_barr_initial < G_barr_threshold:
        raise RuntimeError
    def f(T):
        reweighted_F = reweight_free_energy(x, free_energy, T_sim, T, **kwargs)
        reweighted_F_in_kT = convert_unit(reweighted_F, T=T)
        G_barr = np.max(reweighted_F_in_kT) - reweighted_F_in_kT[np.argmin(x)]
        return G_barr - G_barr_threshold

    for Delta_T in range(0, 50, 10):
        x0 = T_m - Delta_T
        T = newton(f, x0)
        Delta_T_star = T_m - T
        if Delta_T_star > 0:
            break
    else:
        raise RuntimeError(f"Fail to find deltaT")
    return Delta_T_star


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
