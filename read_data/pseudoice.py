# Author: Mian Qin
# Date Created: 6/5/24
from pathlib import Path
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp

from utils import calculate_triangle_area, compute_mean_curvature, format_uncertainty
from utils_plot import create_fig_ax, save_figure, plot_with_error_band, combine_images
from op_dataset import OPDataset, load_dataset
from eda import EDA
from free_energy import SparseSampling, BinlessWHAM
from free_energy_reweighting import reweight_free_energy, get_delta_T_star

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
        column_names=["t", "QBAR", "QBAR_TOP", "box.N", "box.Ntilde", "box_top.N", "box_top.Ntilde",
                      "bias_qbar.value", "bias_qbar_top.value", "chillplus", "lambda_with_PI"],
        column_types={"t": float, "QBAR": float, "chillplus": int, "lambda_with_PI": int},
    )
    # dataset: OPDataset = load_dataset(
    #     data_dir=data_dir,
    #     job_params=job_params,
    #     file_type="out",
    #     column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value"],
    #     column_types={"t": float, "QBAR": float, "box.N": int, "box.Ntilde": float, "bias_qbar.value": float},
    # )
    return dataset


def read_interface(rho, process):
    job_params = _load_params(rho, process)
    dataset = read_data(rho, process)
    interface_type_dict = {0: "IW", 1: "IS"}
    interface_dict = {}

    op_in = ["QBAR", "lambda_with_PI"]
    op_out = "lambda_with_PI"

    for job_name, params in job_params.items():
        interface_dict[job_name] = {}
        data_dir = dataset.data_dir / job_name
        with open(data_dir / "interface.pickle", "rb") as file:
            nodes, faces, interface_type = pickle.load(file)
        for k, v in interface_type_dict.items():
            index = np.where(interface_type == k)
            faces_v = faces[index]
            interface_dict[job_name][v] = [nodes, faces_v]
    return interface_dict


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
    ss.plot_detail(save_dir=figure_save_dir)


def post_processing_wham(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure"
    dataset = read_data(rho, process)

    op_in = ["QBAR", "lambda_with_PI"]
    op_out = "lambda_with_PI"
    wham = BinlessWHAM(dataset, op_in, op_out, bin_range=(10, 580))
    wham.load_result()
    # wham.calculate(with_uncertainties=1, n_iter=1000)
    # wham.save_result()
    wham.plot_free_energy(save_dir=figure_save_dir)

    index = np.where(wham.bin_midpoint > 300)
    x = wham.bin_midpoint[index]
    y = unp.nominal_values(wham.energy)[index]

    p = np.polyfit(x, y, 1)
    print(p)


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


def main_Delta_T_star(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure"
    job_params = _load_params(rho, process)
    dataset = read_data(rho, process)
    interface_type_dict = {0: "IW", 1: "IS"}
    info = {f"A_{v}": [] for v in interface_type_dict.values()}
    info["x_A"] = []

    op_in = ["QBAR", "lambda_with_PI"]
    op_out = "lambda_with_PI"

    for job_name, params in job_params.items():
        data_dir = dataset.data_dir / job_name
        op_data = dataset[job_name]
        x_A = op_data.df_prd[op_out].mean()
        info["x_A"].append(x_A)
        with open(data_dir / "interface.pickle", "rb") as file:
            nodes, faces, interface_type = pickle.load(file)
        for k, v in interface_type_dict.items():
            index = np.where(interface_type == k)
            faces_v = faces[index]
            _, A_v = calculate_triangle_area(nodes, faces_v)  # Unit: Angstrom^2
            A_v = A_v * 1e-20  # to unit m^2
            info[f"A_{v}"].append(A_v)
    for k, v in info.items():
        info[k] = np.array(v)

    wham = BinlessWHAM(dataset, op_in, op_out, bin_range=(10, 550))
    wham.load_result()
    x, F = wham.bin_midpoint, wham.energy
    Delta_T_star = get_delta_T_star(x, F, 300, **info)

    title = fr"Free Energy at Different $\Delta T$, $\Delta T^* = {Delta_T_star:.0f}$ K"
    x_label = fr"$\lambda$"
    y_label = fr"$G(x;\Delta T)$ (kJ/mol)"
    fig, ax = create_fig_ax(title, x_label, y_label)

    T_m = 272
    for Delta_T in range(int(Delta_T_star) - 15, int(Delta_T_star) + 16, 5):
        T = T_m - Delta_T
        label = fr"$\Delta T = {Delta_T}\ \mathrm{{K}}$"
        reweighted_F = reweight_free_energy(x, F, 300, T, **info)
        plot_with_error_band(ax, x, reweighted_F, label=label)

    ax.legend()
    save_path = figure_save_dir / f"free_energy_reweighting.png"
    save_figure(fig, save_path)
    plt.close(fig)


def main_plot_mean_curvature(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure/mean_curvature"
    interface_dict = read_interface(rho, process)
    for job_name in interface_dict.keys():
        x_star = int(job_name.split("_")[1])
        if not 300 <= x_star <= 1200:
            continue
        nodes, faces = interface_dict[job_name]["IW"]
        H = compute_mean_curvature(nodes, faces)

        for i in range(1):
            invalid_index = np.where(np.abs(H - H.mean()) >= 3 * H.std(ddof=1))
            H[invalid_index] = H.mean()  # Values out of 3 sigma are set to mean value

        x = nodes[:, 0]
        y = nodes[:, 1]
        triang = mtri.Triangulation(x, y, faces)

        title = fr"Mean Curvature, ${format_uncertainty(H.mean(), H.std(ddof=1))}$"
        x_label = "$x$"
        y_label = "$y$"
        fig, ax = create_fig_ax(title, x_label, y_label)
        plt.tripcolor(triang, H, shading='flat', cmap='viridis', vmin=-4, vmax=4)
        plt.colorbar(label='Mean Curvature')

        save_figure(fig, figure_save_dir / f"{job_name}.png")
        plt.close(fig)


def main_plot_interface(rho, process):
    figure_save_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}/figure"

    ice_interface_dir = figure_save_dir / "ice_interface"
    image_names = ["op_0300.png", "op_0400.png", "op_0500.png"]
    titles = ["0300", "0400", "0500"]
    image_paths = [ice_interface_dir / image_name for image_name in image_names]
    fig = combine_images(image_paths, titles, 1, 3)
    save_path = figure_save_dir / "ice_interface_combined.png"
    save_figure(fig, save_path)


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
        lambda_chillplus = df["chillplus"].values
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
    # ax.plot(qbar_array, lambda_chillplus_array, "b+", label="lambda_chillplus")
    ax.plot(qbar_array, lambda_with_PI_array, 'b+', label="lambda_with_PI")
    p = np.polyfit(qbar_array, lambda_with_PI_array, 1)
    ax.plot(qbar_array, np.polyval(p, qbar_array), 'r-', label=f"y = {p[0]:.3}x + {p[1]:.3}")

    ax.legend()
    save_path = figure_save_dir / "lambda_qbar.png"
    save_figure(fig, save_path)
    plt.close(fig)


def main():
    process = "melting_270K"
    rho = 0.75
    # post_processing_eda(rho, process)
    # post_processing_ss(rho, process)
    # post_processing_wham(rho, process)
    # main_Delta_T_star(rho, process)
    # main_plot_mean_curvature(rho, process)
    # main_plot_interface(rho, process)
    calc_plot_lambda_q(rho, process)


if __name__ == "__main__":
    main()
