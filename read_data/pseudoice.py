# Author: Mian Qin
# Date Created: 6/5/24
from pathlib import Path
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import uncertainties.unumpy as unp

from utils import convert_unit, read_solid_like_atoms
from utils_plot import create_fig_ax, save_figure, plot_with_error_bar, plot_with_error_band
from op_dataset import OPDataset, load_dataset
from eda import EDA
from sparse_sampling import SparseSampling
from wham import BinlessWHAM

run_env = os.environ.get("RUN_ENVIRONMENT")
if run_env == "wsl":
    home_path = Path("/home/qinmian")
elif run_env == "chestnut":
    home_path = Path("/home/mianqin")
elif run_env == "mianqin_PC":
    home_path = Path("//wsl.localhost/Ubuntu-22.04/home/qinmian")
else:
    raise RuntimeError(f"Unknown environment: {run_env}")


def _load_params(rho, process) -> dict:
    data_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}"
    with open(data_dir / "job_params.json", 'r') as file:
        job_params = json.load(file)
    return job_params


def read_data(rho, process) -> OPDataset:
    data_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/result"
    job_params = _load_params(rho, process)

    dataset: OPDataset = load_dataset(
        data_dir=data_dir,
        job_params=job_params,
        file_type="csv",
        column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value", "lambda_with_PI", "lambda_chillplus"],
        column_types={"t": float, "QBAR": float, "box.N": int, "box.Ntilde": float, "bias_qbar.value": float,
                      "lambda_with_PI": int, "lambda_chillplus": int},
    )
    # dataset: OPDataset = load_dataset(
    #     data_dir=data_dir,
    #     job_params=job_params,
    #     file_type="out",
    #     column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value"],
    #     column_types={"t": float, "QBAR": float, "box.N": int, "box.Ntilde": float, "bias_qbar.value": float},
    # )
    return dataset


def post_processing_eda(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure"
    op = "QBAR"
    dataset = read_data(rho, process)

    eda = EDA(dataset, op, figure_save_dir)
    eda.plot_ddF(figure_save_dir)
    eda.determine_relaxation_time()
    eda.calculate_acf()
    eda.determine_autocorr_time(figure_save_dir, ignore_previous=0)
    eda.plot_op(save_dir=figure_save_dir)
    eda.plot_histogram(bin_width=2, bin_range=(0, 600), save_dir=figure_save_dir)
    eda.plot_acf(save_dir=figure_save_dir)
    eda.plot_act(save_dir=figure_save_dir)


def post_processing_ss(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure"
    op = "QBAR"
    dataset = read_data(rho, process)

    ss = SparseSampling(dataset, op)
    ss.calculate()
    ss.plot_free_energy(save_dir=figure_save_dir)
    ss.plot_different_DeltaT(save_dir=figure_save_dir)
    ss.plot_detail(save_dir=figure_save_dir)


def post_processing_wham(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure"
    dataset = read_data(rho, process)

    op_in = ["QBAR", "lambda_with_PI"]
    op_out = "lambda_with_PI"
    wham = BinlessWHAM(dataset, op_in, op_out, bin_range=(10, 550))
    wham.load_result()
    # wham.calculate(with_uncertainties=1, n_iter=1000)
    # wham.save_result()
    wham.plot_free_energy(save_dir=figure_save_dir)
    wham.plot_different_DeltaT(save_dir=figure_save_dir)

    # Find the location of the maximum slope
    # x = wham.bin_midpoint
    # y_u = convert_unit(wham.energy)
    # y = unp.nominal_values(y_u)
    # s_y = unp.std_devs(y_u)
    #
    # def f(t, a4, a3, a2, a1, a0):
    #     return np.polyval([a4, a3, a2, a1, a0], t)
    #
    # def df(t, a4, a3, a2, a1, a0):
    #     return np.polyval([4 * a4, 3 * a3, 2 * a2, 1 * a1], t)
    #
    # p_opt, p_cov = curve_fit(f, x, y, sigma=s_y, absolute_sigma=True)
    # a4, a3, a2, a1, a0 = p_opt
    #
    # x_plot = np.linspace(x.min(), x.max(), 1000)
    # y_plot = np.polyval(p_opt, x_plot)
    #
    # title = "Free Energy"
    # x_label = r"$\lambda$"
    # y_label = r"$\beta F$"
    # fig, ax = create_fig_ax(title, x_label, y_label)
    #
    # wham.plot_free_energy_plot_line(ax)
    # ax.plot(x_plot, y_plot, "r--", label=fr"y = {a4:.3g}x^4 + {a3:.3g}x^3 + {a2:.3g}x^2 + {a1:.3g}x + {a0:.3g}")
    # ax.legend()
    #
    # save_path = figure_save_dir / f"wham_free_energy_fitting.png"
    # save_figure(fig, save_path)
    # plt.close(fig)
    #
    # title = "Derivative of Free Energy"
    # x_label = r"$\lambda$"
    # y_label = r"$\beta dF/d\lambda$"
    # fig, ax = create_fig_ax(title, x_label, y_label)
    #
    # dy_dx_plot = df(x_plot, *p_opt)
    #
    # ax.plot(x_plot, dy_dx_plot, "r--")
    #
    # save_path = figure_save_dir / f"wham_derivative.png"
    # save_figure(fig, save_path)
    # plt.close(fig)


def calc_plot_lambda_q(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure"
    dataset = read_data(rho, process)

    qbar_list = []
    lambda_chillplus_list = []
    lambda_with_PI_list = []
    for job_name, op_data in dataset.items():
        df = op_data.df_prd
        qbar = df["QBAR"].values
        lambda_with_PI = df["lambda_with_PI"].values
        lambda_chillplus = df["lambda_chillplus"].values
        qbar_list.append(qbar)
        lambda_chillplus_list.append(lambda_chillplus)
        lambda_with_PI_list.append(lambda_with_PI)

    qbar_array = np.concatenate(qbar_list)
    lambda_chillplus_array = np.concatenate(lambda_chillplus_list)
    lambda_with_PI_array = np.concatenate(lambda_with_PI_list)

    title = rf"Number of Ice-like Water, $\rho = {rho}$, {process}"
    x_label = "qbar"
    y_label = r"$\lambda$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    ax.plot(qbar_array, lambda_chillplus_array, "b+", label="lambda_chillplus")
    ax.plot(qbar_array, lambda_with_PI_array, 'g+', label="lambda_with_PI")

    ax.legend()
    save_path = figure_save_dir / "lambda_qbar.png"
    save_figure(fig, save_path)
    plt.close(fig)


def linear(x, k, b):
    return k * x + b


def plot_difference(rho):
    figure_save_dir = home_path / f"/home/qinmian/data/gromacs/pseudoice/figure"
    op = "QBAR"
    process_list = ["melting_300K", "melting_270K"]

    title = fr"Free Energy Difference, $\alpha = {rho}$, Ref: {process_list[0]}"
    x_label = "lambda"
    y_label = fr"$F$ (kJ/mol)"
    fig, ax = create_fig_ax(title, x_label, y_label)
    fig2, ax2 = create_fig_ax("Free Energy $-$ Linear Fit", x_label, y_label)

    process_ref = process_list[0]
    dataset = read_data(rho, process_ref)
    # ss = SparseSampling(dataset, op)
    # ss.calculate()
    # x_u_ref = ss.x_u
    # x_ref = unp.nominal_values(x_u_ref)
    # F_nu_u_ref = unp.nominal_values(ss.F_nu_u)
    op_in = ["QBAR", "lambda_with_PI"]
    op_out = "lambda_with_PI"
    wham = BinlessWHAM(dataset, op_in, op_out)
    wham.load_result()
    x_ref = wham.bin_midpoint
    F_u_ref = wham.energy
    F_ref = unp.nominal_values(F_u_ref)
    s_F_ref = unp.std_devs(F_u_ref)
    for process in process_list[1:]:
        dataset = read_data(rho, process)
        # ss = SparseSampling(dataset, op)
        # ss.calculate()
        # x_u = ss.x_u
        # x = unp.nominal_values(x_u)
        # F_nu_u = unp.nominal_values(ss.F_nu_u)
        wham = BinlessWHAM(dataset, op_in, op_out)
        wham.load_result()
        x = wham.bin_midpoint
        F_u = wham.energy
        F = unp.nominal_values(F_u)
        s_F = unp.std_devs(F_u)
        x_plot = np.linspace(max(x_ref.min(), x.min()), min(x_ref.max(), x.max()), 1000)
        y_interp = np.interp(x_plot, x, F)
        s_y_interp = np.interp(x_plot, x, s_F)
        y_interp_ref = np.interp(x_plot, x_ref, F_ref)
        s_y_interp_ref = np.interp(x_plot, x_ref, s_F_ref)
        y_interp_u = unp.uarray(y_interp, s_y_interp)
        y_interp_ref_u = unp.uarray(y_interp_ref, s_y_interp_ref)
        y_plot_u = y_interp_u - y_interp_ref_u
        y_plot = unp.nominal_values(y_plot_u)
        s_y_plot = unp.std_devs(y_plot_u)
        # Fit data where
        index = (x_plot > 1500) & (x_plot < 2800)
        p, p_cov = curve_fit(linear, x_plot[index], y_plot[index], sigma=s_y_plot[index], absolute_sigma=True)
        s_p = np.sqrt(np.diag(p_cov))
        p_u = unp.uarray(p, s_p)
        y_fit = linear(x_plot, *p_u)
        delta_y_u = y_plot_u - y_fit
        plot_with_error_band(ax, x_plot, y_plot_u, "-", label=f"{process}")
        plot_with_error_band(ax, x_plot, y_fit, "--",
                             label=fr"linear fit of {process}, $y = ({p[0]:.4f}\pm{s_p[0]:.4f})x + ({p[1]:.1f}\pm{s_p[1]:.1f})$")
        plot_with_error_band(ax2, x_plot, delta_y_u, "-", label=f"{process}")
    ax.legend()
    ax2.legend()
    save_path = figure_save_dir / f"energy_difference_{rho}.png"
    save_figure(fig, save_path)
    save_path = figure_save_dir / f"energy_difference_minus_linear_fit_{rho}.png"
    save_figure(fig2, save_path)


def compare_free_energy(rho):
    figure_save_dir = home_path / f"/home/qinmian/data/gromacs/pseudoice/figure"

    op = "QBAR"
    # process_list = ["melting_270K", "melting_300K"]
    process_list = ["melting_300K", "icing_300K_20p1ns", "icing_300K", "icing_constant_ramp_rate"]
    ss_list = []
    for process in process_list:
        dataset = read_data(rho, process)
        ss = SparseSampling(dataset, op)
        ss.calculate()
        ss_list.append(ss)

    title = fr"Comparison of Free Energy, $\alpha = {rho}$"
    x_label = "qbar"
    y_label = fr"$\beta F$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    for process, ss in zip(process_list, ss_list):
        ss.plot_free_energy_plot_line(ax, label=process)
    ax.legend()
    save_path = figure_save_dir / f"comparison_{rho}.png"
    save_figure(fig, save_path)

    title = fr"Comparison of $\beta dF_{{\lambda}} / dx$, $\alpha = {rho}$"
    x_label = "qbar"
    y_label = r"$\beta dF_{\lambda} / dx$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    for process, ss in zip(process_list, ss_list):
        ss.plot_dF_lambda_dx_star_plot_line(ax, label=process)
    ax.legend()
    save_path = figure_save_dir / f"comparison_detail_{rho}.png"
    save_figure(fig, save_path)


def main():
    process = "melting_300K"
    rho = 0.75
    # post_processing_eda(rho, process)
    # post_processing_ss(rho, process)
    post_processing_wham(rho, process)
    # calc_plot_lambda_q(rho, process)

    # for rho in [1.0]:
    #     compare_free_energy(rho)
        # plot_difference(rho)


if __name__ == "__main__":
    main()
