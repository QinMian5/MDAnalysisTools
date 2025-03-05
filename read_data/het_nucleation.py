# Author: Mian Qin
# Date Created: 2025/1/15
from pathlib import Path
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

from utils import calculate_triangle_area
from utils_plot import create_fig_ax, save_figure, plot_with_error_bar, plot_with_error_band, combine_images
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
    data_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}"
    with open(data_dir / "result" / "job_params.json", 'r') as file:
        job_params = json.load(file)
    return job_params


def read_data(rho, process) -> OPDataset:
    data_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/result"
    job_params = _load_params(rho, process)

    if process == "melting_300K":
        dataset: OPDataset = load_dataset(
            data_dir=data_dir,
            job_params=job_params,
            file_type="csv",
            column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value", "lambda_chillplus", "lambda_with_PI"],
            column_types={"t": float, "QBAR": float, "box.N": int, "box.Ntilde": float, "bias_qbar.value": float,
                          "lambda_chillplus": int, "lambda_with_PI": int},
        )
    elif process == "melting_270K":
        dataset: OPDataset = load_dataset(
            data_dir=data_dir,
            job_params=job_params,
            file_type="csv",
            column_names=["t", "QBAR", "box_inner.N", "box_inner.Ntilde", "box_outer.N", "box_outer.Ntilde",
                          "bias_qbar.value", "bias_phi.value", "chillplus", "lambda_with_PI"],
            column_types={"t": float, "QBAR": float, "chillplus": int, "lambda_with_PI": int},
        )
    return dataset


def get_A_mean(rho, process):
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
    return info


def get_A_instantaneous(rho, process):
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
        with open(data_dir / "instantaneous_interface.pickle", "rb") as file:
            instantaneous_interface_dict = pickle.load(file)

        for k, v in interface_type_dict.items():
            A_v_list = []
            for t, [nodes, faces, interface_type] in instantaneous_interface_dict.items():
                index = np.where(interface_type == k)
                faces_v = faces[index]
                _, A_v = calculate_triangle_area(nodes, faces_v)  # Unit: Angstrom^2
                A_v = A_v * 1e-20  # to unit m^2
                A_v_list.append(A_v)
            A_v_list = np.array(A_v_list)
            info[f"A_{v}"].append(np.mean(A_v_list))
            # u_A_v = ufloat(np.mean(A_v_list), np.std(A_v_list, ddof=1) / np.sqrt(len(A_v_list)))
            # info[f"A_{v}"].append(u_A_v)
    for k, v in info.items():
        info[k] = np.array(v)
    return info


def extrapolate_A(info):
    def f_A_IW(x, a_IW):
        return a_IW * x ** (2/3)

    def f_A_IS(x, a_IS):
        return a_IS * x ** (2/3)

    x_A = info["x_A"]
    A_IW = info["A_IW"]
    threshold = 100e-20
    index_to_fit = np.where(A_IW >= threshold)
    index_to_extrapolate = np.where(A_IW < threshold)
    A_IW_original = A_IW[index_to_fit]
    popt, pcov = curve_fit(f_A_IW, x_A[index_to_fit], A_IW_original)
    A_IW_extrapolate = f_A_IW(x_A[index_to_extrapolate], *popt)
    A_IW_info = [[x_A[index_to_fit], A_IW_original], [x_A[index_to_extrapolate], A_IW_extrapolate]]

    A_IS = info["A_IS"]
    index_to_fit = np.where(A_IS >= threshold)
    index_to_extrapolate = np.where(A_IS < threshold)
    A_IS_original = A_IS[index_to_fit]
    popt, pcov = curve_fit(f_A_IS, x_A[index_to_fit], A_IS_original)
    A_IS_extrapolate = f_A_IS(x_A[index_to_extrapolate], *popt)
    A_IS_info = [[x_A[index_to_fit], A_IS_original], [x_A[index_to_extrapolate], A_IS_extrapolate]]
    return A_IW_info, A_IS_info


def post_processing_eda(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
    op = "QBAR"
    dataset = read_data(rho, process)

    eda = EDA(dataset, op, figure_save_dir)
    eda.plot_ddF(figure_save_dir)
    eda.determine_relaxation_time()
    eda.calculate_acf()
    eda.determine_autocorr_time(figure_save_dir, ignore_previous=0)
    eda.plot_op(save_dir=figure_save_dir)
    eda.plot_histogram(bin_width=5, bin_range=(0, 1000), save_dir=figure_save_dir)
    eda.plot_acf(save_dir=figure_save_dir)
    eda.plot_act(save_dir=figure_save_dir)


def post_processing_ss(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
    op = "QBAR"
    dataset = read_data(rho, process)

    ss = SparseSampling(dataset, op)
    ss.calculate()
    ss.plot_free_energy(save_dir=figure_save_dir)
    # ss.plot_different_DeltaT(save_dir=figure_save_dir)
    ss.plot_detail(save_dir=figure_save_dir)
    ss.save_result()


def post_processing_wham(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
    dataset = read_data(rho, process)

    op_in = ["QBAR", "lambda_with_PI"]
    op_out = "lambda_with_PI"
    wham = BinlessWHAM(dataset, op_in, op_out, bin_range=(10, 380))
    # wham.load_result()
    wham.calculate(with_uncertainties=1, n_iter=1000)
    wham.save_result()
    wham.plot_free_energy(save_dir=figure_save_dir)


def geometric_sphere_fit(points):
    """Analytic sphere fitting using 4 points"""
    A = np.zeros((4, 4))
    A[:, 0] = points[:, 0]
    A[:, 1] = points[:, 1]
    A[:, 2] = points[:, 2]
    A[:, 3] = 1
    f = np.sum(points ** 2, axis=1)
    c = np.linalg.solve(A, f)
    center = c[:3] / 2
    radius = np.sqrt(c[3] + np.sum(center ** 2))
    return center, radius


def fit_spherical_cap(nodes: np.ndarray, n_ransac=1000, eps=1.0, delta_z=0.5):
    if len(nodes) == 0:
        return {"plane_z": 0,
                "plane_z_std": 0,
                "sphere_center": 0,
                "sphere_center_std": 0,
                "sphere_radius": 0,
                "sphere_radius_std": 0,
                "contact_angle": 0,
                "contact_angle_std": 0}
    z = nodes[:, 2]
    hist, bin_edges = np.histogram(z, bins="auto")
    max_bin = np.argmax(hist)
    z_low, z_high = bin_edges[max_bin], bin_edges[max_bin + 1]

    plane_mask = (z >= z_low - delta_z * (z_high - z_low)) & \
                 (z <= z_high + delta_z * (z_high - z_low))
    plane_nodes = nodes[plane_mask]

    z0 = np.mean(plane_nodes[:, 2])
    z0_std = np.std(plane_nodes[:, 2]) / np.sqrt(len(plane_nodes))

    # ================== Sphere Fitting ================== #
    # Exclude plane region nodes
    sphere_nodes = nodes[~plane_mask]
    if len(sphere_nodes) < 4:
        raise ValueError("Insufficient non-planar nodes for sphere fitting")

    # RANSAC phase
    best_params, max_inliers = None, 0
    for _ in range(n_ransac):
        sample = sphere_nodes[np.random.choice(len(sphere_nodes), 4, replace=False)]

        # Fit sphere through samples
        try:
            center, radius = geometric_sphere_fit(sample)
        except np.linalg.LinAlgError:
            continue

        # Find inliers
        distances = np.linalg.norm(sphere_nodes - center, axis=1)
        inliers = np.abs(distances - radius) < eps
        n_inliers = np.sum(inliers)
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_params = (center, radius, inliers)

    # Refine with the Least Squares Method
    def residual(params):
        center, radius = params[:3], params[3]
        return np.linalg.norm(sphere_nodes - center, axis=1) - radius

    center_init, radius_init, inlier_mask = best_params
    inlier_nodes = sphere_nodes[inlier_mask]
    opt_result = optimize.least_squares(
        residual,
        x0=np.append(center_init, radius_init),
        loss="soft_l1"
    )
    center_opt = opt_result.x[:3]
    radius_opt = opt_result.x[3]

    # Uncertainty estimation
    residuals = residual(opt_result.x)
    radius_std = np.std(residuals)
    jac = opt_result.jac
    cov_matrix = np.linalg.inv(jac.T @ jac) * (radius_std ** 2)
    center_std = np.sqrt(np.diag(cov_matrix))[:3]

    # =============== Contact Angle Calculation =============== #
    dz = center_opt[2] - z0
    if abs(dz) > radius_opt:
        raise ValueError("Invalid spherical cap configuration")

    theta = np.arccos(dz / radius_opt)

    # Error propagation
    var_zc = center_std[2] ** 2
    var_z0 = z0_std ** 2
    var_r = radius_std ** 2

    dtheta_dzc = -1 / (radius_opt * np.sin(theta))
    dtheta_dz0 = 1 / (radius_opt * np.sin(theta))
    dtheta_dr = -dz / (radius_opt ** 2 * np.sin(theta))

    # Check uncertainty calculation
    # z0_u = ufloat(z0, z0_std)
    # z_u = ufloat(center_opt[2], center_std[2])
    # dz = z_u - z0_u
    # radius_u = ufloat(radius_opt, radius_std)
    # theta_u = unp.arccos(dz / radius_u)

    theta_std = np.sqrt(
        (dtheta_dzc ** 2 * var_zc) +
        (dtheta_dz0 ** 2 * var_z0) +
        (dtheta_dr ** 2 * var_r)
    )
    print(f"{np.degrees(theta):.2f} +- {np.degrees(theta_std):.2f}")

    results = {
        "plane_z": z0,
        "plane_z_std": z0_std,
        "sphere_center": center_opt,
        "sphere_center_std": center_std,
        "sphere_radius": radius_opt,
        "sphere_radius_std": radius_std,
        "contact_angle": theta,
        "contact_angle_std": theta_std,
    }
    return results


def read_fitting_result(rho, process):
    job_params = _load_params(rho, process)
    dataset = read_data(rho, process)
    op = "QBAR"
    x_star_list = []
    sphere_radius_list = []
    sphere_radius_std_list = []
    contact_angle_list = []
    contact_angle_std_list = []
    for job_name, params in job_params.items():
        x_star_list.append(params[op]["X_STAR"])
        intermediate_data_dir = dataset.data_dir.parent / "intermediate_result" / job_name
        with open(intermediate_data_dir / "spherical_cap_fitting.pickle", "rb") as file:
            fitting_results = pickle.load(file)
        sphere_radius = fitting_results["sphere_radius"]
        sphere_radius_std = fitting_results["sphere_radius_std"]
        contact_angle = fitting_results["contact_angle"]
        contact_angle_std = fitting_results["contact_angle_std"]
        sphere_radius_list.append(sphere_radius)
        sphere_radius_std_list.append(sphere_radius_std)
        contact_angle_list.append(contact_angle)
        contact_angle_std_list.append(contact_angle_std)
    x_star_array = np.array(x_star_list)
    sphere_radius_u = unp.uarray(sphere_radius_list, sphere_radius_std_list)
    contact_angle_u = unp.uarray(contact_angle_list, contact_angle_std_list)
    return x_star_array, sphere_radius_u, contact_angle_u


def post_processing_fit_spherical_cap(rho, process):
    job_params = _load_params(rho, process)
    dataset = read_data(rho, process)
    for job_name, params in job_params.items():
        data_dir = dataset.data_dir / job_name
        save_dir = dataset.data_dir.parent / "intermediate_result" / job_name
        with open(data_dir / "interface.pickle", "rb") as file:
            nodes, faces, interface_type = pickle.load(file)
        results = fit_spherical_cap(nodes)
        with open(save_dir / "spherical_cap_fitting.pickle", "wb") as file:
            pickle.dump(results, file)


def main_plot_R_theta(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
    x_star_array, sphere_radius_u, contact_angle_u = read_fitting_result(rho, process)

    title = "Fitting Sphere Radius"
    x_label = r"$x^*$"
    y_label = r"$r$ (Å)"
    fig, ax = create_fig_ax(title, x_label, y_label)
    plot_with_error_band(ax, x_star_array, sphere_radius_u, "go-")
    save_path = figure_save_dir / f"fitting_sphere_radius.png"
    save_figure(fig, save_path)
    plt.close(fig)

    title = "Fitting Contact Angle"
    x_label = r"$x^*$"
    y_label = r"$\theta$ (degree)"
    fig, ax = create_fig_ax(title, x_label, y_label)
    plot_with_error_band(ax, x_star_array, contact_angle_u * 180 / np.pi, "bo-")
    save_path = figure_save_dir / f"fitting_contact_angle.png"
    save_figure(fig, save_path)
    plt.close(fig)


def main_plot_A_mean_vs_instantaneous(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
    info_mean = get_A_mean(rho, process)
    info_instantaneous = get_A_instantaneous(rho, process)

    title = "Mean vs Instantaneous"
    x_label = r"$\lambda$"
    y_label = r"Surface Area $(m^2)$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    color_IW = ax.plot(info_mean["x_A"], info_mean["A_IW"], "o-", label=r"$A_{iw}$ mean")[0].get_color()
    color_IS = ax.plot(info_mean["x_A"], info_mean["A_IS"], "o-", label=r"$A_{is}$ mean")[0].get_color()
    plot_with_error_band(ax, info_instantaneous["x_A"], info_instantaneous["A_IW"], "o--", color=color_IW, label=r"$A_{iw}$ instantaneous")
    plot_with_error_band(ax, info_instantaneous["x_A"], info_instantaneous["A_IS"], "o--", color=color_IS, label=r"$A_{is}$ instantaneous")
    ax.legend()

    save_path = figure_save_dir / f"mean_vs_instantaneous.png"
    save_figure(fig, save_path)
    plt.close(fig)


def main_plot_extrapolation(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
    info = get_A_mean(rho, process)
    A_IW_info, A_IS_info = extrapolate_A(info)

    title = "Extrapolation"
    x_label = r"$\lambda$"
    y_label = r"Surface Area $(m^2)$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    color_IW = ax.plot(info["x_A"], info["A_IW"], "o-", label=r"original $A_{iw}$")[0].get_color()
    ax.plot(*A_IW_info[1], "o--", color=color_IW, label=r"extrapolated $A_{iw}$")
    color_IS = ax.plot(info["x_A"], info["A_IS"], "o-", label=r"original $A_{is}$")[0].get_color()
    ax.plot(*A_IS_info[1], "o--", color=color_IS, label=r"extrapolated $A_{is}$")
    ax.legend()

    save_path = figure_save_dir / f"extrapolation.png"
    save_figure(fig, save_path)
    plt.close(fig)


def compare_with_theory():
    rho = 0.75
    process = "melting_300K"
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"

    x_star_array, sphere_radius_u, contact_angle_u = read_fitting_result(rho, process)
    info = get_A_mean(rho, process)
    x_A = info["x_A"]
    A_IW = info["A_IW"]
    A_IS = info["A_IS"]
    A_IW_theory = 2 * np.pi * sphere_radius_u ** 2 * (1 - unp.cos(contact_angle_u))
    A_IS_theory = np.pi * sphere_radius_u ** 2 * unp.sin(contact_angle_u) ** 2

    title = "Surface Area Comparison"
    x_label = r"$x^*$"
    y_label = r"Surface Area $(Å^2)$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    color_IW = ax.plot(x_A, A_IW, "o-", label=r"$A_{iw}$")[0].get_color()
    color_IS = ax.plot(x_A, A_IS, "o-", label=r"$A_{is}$")[0].get_color()
    plot_with_error_band(ax, x_A, A_IW_theory, label=r"$A_{iw, \mathrm{theory}}$", color=color_IW, linestyle="--")
    plot_with_error_band(ax, x_A, A_IS_theory, label=r"$A_{is, \mathrm{theory}}$", color=color_IS, linestyle="--")
    ax.legend()

    save_path = figure_save_dir / "surface_area_comparison.png"
    save_figure(fig, save_path)
    plt.close(fig)


def compare_reweighting_method():
    rho = 0.75
    process_270K = "melting_270K"
    process_300K = "melting_300K"
    op = "QBAR"

    dataset_270K = read_data(rho, process_270K)
    dataset_300K = read_data(rho, process_300K)
    # info_270K = get_A_instantaneous(rho, process_270K)
    # info_300K = get_A_instantaneous(rho, process_300K)
    info_270K = get_A_mean(rho, process_270K)
    info_300K = get_A_mean(rho, process_300K)

    op_in = ["QBAR", "lambda_with_PI"]
    op_out = "lambda_with_PI"
    wham_270K = BinlessWHAM(dataset_270K, op_in, op_out, bin_range=(10, 380))
    wham_270K.load_result()
    wham_300K = BinlessWHAM(dataset_300K, op_in, op_out, bin_range=(10, 380))
    wham_300K.load_result()
    x_270K, F_270K = wham_270K.bin_midpoint, wham_270K.energy
    x_300K, F_300K = wham_300K.bin_midpoint, wham_300K.energy

    # reweighted_F = reweight_free_energy(x_300K, F_300K, 300, 270)  # no surface energy correction
    reweighted_F = reweight_free_energy(x_300K, F_300K, 300, 270, **info_300K)
    real_F = F_270K
    x_align = 100
    index_real = np.argmin(np.abs(x_270K - x_align))
    index_reweighted = np.argmin(np.abs(x_300K - x_align))
    reweighted_F += real_F[index_real] - reweighted_F[index_reweighted]

    title = "Comparison of Reweighted G and Real G"
    x_label = r"$\lambda$"
    y_label = r"$G$ (kJ/mol)"
    fig, ax = create_fig_ax(title, x_label, y_label)
    # plot_with_error_bar(ax, x_270K, F_270K, label="270 K")
    plot_with_error_band(ax, x_270K, real_F, label="Real G")
    plot_with_error_band(ax, x_300K, reweighted_F, label="Reweighted G")
    ax.legend()

    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/figure/comparision_reweighting_simulation.png"
    save_figure(fig, figure_save_dir)
    plt.close(fig)

    # cond_fit_reweighted = x_300K > 100
    # k_reweighted = np.polyfit(x_300K[cond_fit_reweighted], unp.nominal_values(reweighted_F)[cond_fit_reweighted], 1)[0]
    # cond_fit_real = x_270K > 100
    # k_real = np.polyfit(x_270K[cond_fit_real], unp.nominal_values(real_F)[cond_fit_real], 1)[0]
    # print(f"Slope of reweighted G: {k_reweighted:.3f}\n"
    #       f"Slope of real G: {k_real:.3f}")
    # print(f"Correction for DeltaMu: {(k_real - k_reweighted) / 30:.5f} (kJ/mol.K)")


def main_Delta_T_star(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
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
    for Delta_T in range(int(np.round(Delta_T_star)) - 15, int(np.round(Delta_T_star)) + 16, 5):
        T = T_m - Delta_T
        label = fr"$\Delta T = {Delta_T}\ \mathrm{{K}}$"
        reweighted_F = reweight_free_energy(x, F, 300, T, **info)
        plot_with_error_band(ax, x, reweighted_F, label=label)

    ax.legend()
    save_path = figure_save_dir / f"free_energy_reweighting.png"
    save_figure(fig, save_path)
    plt.close(fig)


def main_plot_interface(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"

    ice_interface_dir = figure_save_dir / "ice_interface"
    image_names = ["op_0120.png", "op_0180.png", "op_0240.png", "op_0285.png", "op_0330.png", "op_0390.png"]
    titles = ["0120", "0180", "0240", "0285", "0330", "0390"]
    image_paths = [ice_interface_dir / image_name for image_name in image_names]
    fig = combine_images(image_paths, titles, 2, 3)
    save_path = figure_save_dir / "ice_interface_combined.png"
    save_figure(fig, save_path)


def main():
    rho = 0.75
    process = "melting_270K"

    # post_processing_eda(rho, process)
    post_processing_ss(rho, process)
    # post_processing_wham(rho, process)
    # post_processing_fit_spherical_cap(rho, process)
    # calc_plot_lambda_q(rho, process)
    # compare_reweighting_method()
    # compare_with_theory()
    # main_plot_R_theta(rho, process)
    # main_plot_extrapolation(rho, process)
    # main_plot_A_mean_vs_instantaneous(rho, process)
    # main_Delta_T_star(rho, process)
    # main_plot_interface(rho, process)


if __name__ == "__main__":
    main()
