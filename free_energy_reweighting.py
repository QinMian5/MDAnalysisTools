# Author: Mian Qin
# Date Created: 2025/1/20
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
from scipy.optimize import newton
from scipy.interpolate import interp1d
from scipy.misc import derivative
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


def find_energy_barrier(x, y):
    n = len(x)
    if n == 0:
        raise ValueError

    min_index = 0
    max_barrier = 0

    min_point = (x[0], y[0])
    max_point = (x[0], y[0])

    for i in range(1, n):
        barrier = y[i] - y[min_index]
        if barrier > max_barrier:
            max_barrier = barrier
            min_point = (x[min_index], y[min_index])
            max_point = (x[i], y[i])

        if y[i] < y[min_index]:
            min_index = i

    return max_barrier, min_point, max_point


def get_delta_T_star(x: np.ndarray, free_energy: np.ndarray, T_sim: float, J_target=1e28, **kwargs):
    free_energy = unp.nominal_values(free_energy)
    T_m = T_m_tip4p_ice.value

    def ln_D(T):
        return np.log(D_real_water_PL(T))

    def calc_G_crit(T):
        reweighted_F = reweight_free_energy(x, free_energy, T_sim, T, **kwargs)
        reweighted_F_in_kT = convert_unit(reweighted_F, T=T)
        result = find_energy_barrier(x, reweighted_F_in_kT)
        G_crit = result[0]
        return G_crit

    def calc_G_diff(T):
        G_diff = derivative(ln_D, T, dx=1e-5) * T
        return G_diff

    def f(T):
        n_l = rho_I_tip4p_ice(T) * c.N_A
        G_diff = calc_G_diff(T)
        G_crit = calc_G_crit(T)
        J = c.k * T / c.h * n_l * np.exp(-(G_diff + G_crit))
        log_diff = np.log(J) - np.log(J_target)
        return log_diff

    # Get an initial guess
    DeltaT_trial = np.linspace(0, 50, 50)
    T_trial = T_m - DeltaT_trial
    f_value = np.array([f(T) for T in T_trial])
    f_value_abs = np.abs(f_value)
    index = np.argmin(f_value_abs)
    x0 = T_trial[index]
    T = newton(f, x0)
    Delta_T_star = T_m - T

    G_diff = calc_G_diff(T)
    G_crit = calc_G_crit(T)
    print(f"G_diff = {G_diff}")
    print(f"G_crit = {G_crit}")
    print(f"Delta_T_star = {Delta_T_star}")
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
