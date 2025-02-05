# Author: Mian Qin
# Date Created: 2025/1/15
from pathlib import Path
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
from uncertainties import ufloat
import uncertainties.unumpy as unp

from utils import calculate_triangle_area
from utils_plot import create_fig_ax, save_figure, plot_with_error_bar, plot_with_error_band
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

    dataset: OPDataset = load_dataset(
        data_dir=data_dir,
        job_params=job_params,
        file_type="csv",
        column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value", "lambda_chillplus", "lambda_with_PI"],
        column_types={"t": float, "QBAR": float, "box.N": int, "box.Ntilde": float, "bias_qbar.value": float,
                      "lambda_chillplus": int, "lambda_with_PI": int},
    )
    return dataset


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
    wham = BinlessWHAM(dataset, op_in, op_out, bin_range=(10, 390))
    # wham.load_result()
    wham.calculate(with_uncertainties=1, n_iter=1000)
    wham.save_result()
    wham.plot_free_energy(save_dir=figure_save_dir)
    # wham.plot_different_DeltaT(save_dir=figure_save_dir)


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


def compare_with_theory():
    rho = 0.75
    process = "melting_300K"
    op = "QBAR"
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"

    job_params = _load_params(rho, process)
    dataset = read_data(rho, process)
    interface_type_dict = {0: "IW", 1: "IS"}
    info = {f"A_{v}": [] for v in interface_type_dict.values()}
    x_star_list = []
    sphere_radius_list = []
    sphere_radius_std_list = []
    contact_angle_list = []
    contact_angle_std_list = []
    for job_name, params in job_params.items():
        x_star_list.append(params[op]["X_STAR"])
        data_dir = dataset.data_dir / job_name
        intermediate_data_dir = dataset.data_dir.parent / "intermediate_result" / job_name
        with open(data_dir / "interface.pickle", "rb") as file:
            nodes, faces, interface_type = pickle.load(file)
        for k, v in interface_type_dict.items():
            index = np.where(interface_type == k)
            faces_v = faces[index]
            _, A_v = calculate_triangle_area(nodes, faces_v)  # Unit: Angstrom^2
            # A_v = A_v * 1e-20  # to unit m^2
            info[f"A_{v}"].append(A_v)
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
    for k, v in info.items():
        info[k] = np.array(v)
    sphere_radius_u = unp.uarray(sphere_radius_list, sphere_radius_std_list)
    contact_angle_u = unp.uarray(contact_angle_list, contact_angle_std_list)
    A_IW = info["A_IW"]
    A_IS = info["A_IS"]
    A_IW_theory = 2 * np.pi * sphere_radius_u ** 2 * (1 - unp.cos(contact_angle_u))
    A_IS_theory = np.pi * sphere_radius_u ** 2 * unp.sin(contact_angle_u) ** 2

    title = "Surface Area Comparison"
    x_label = r"$x^*$"
    y_label = r"$\mathrm{Surface Area}\ (Å^2)$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    color_IW = ax.plot(x_star_array, A_IW, "o-", label=r"$A_{iw}$")[0].get_color()
    color_IS = ax.plot(x_star_array, A_IS, "o-", label=r"$A_{is}$")[0].get_color()
    plot_with_error_band(ax, x_star_array, A_IW_theory, label=r"$A_{iw, \mathrm{theory}}$", color=color_IW, linestyle="--")
    plot_with_error_band(ax, x_star_array, A_IS_theory, label=r"$A_{is, \mathrm{theory}}$", color=color_IS, linestyle="--")
    ax.legend()

    save_path = figure_save_dir / "surface_area_comparison.png"
    save_figure(fig, save_path)
    plt.close(fig)


def compare_reweighting_method():
    rho = 0.75
    # process_270K = "melting_270K"
    process_300K = "melting_300K"
    op = "QBAR"

    job_params_300K = _load_params(rho, process_300K)
    # dataset_270K = read_data(rho, process_270K)
    dataset_300K = read_data(rho, process_300K)
    interface_type_dict = {0: "IW", 1: "IS"}
    info = {f"A_{v}": [] for v in interface_type_dict.values()}

    for job_name, params in job_params_300K.items():
        data_dir = dataset_300K.data_dir / job_name
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

    # ss_270K = SparseSampling(dataset_270K, op)
    ss_300K = SparseSampling(dataset_300K, op)
    # ss_270K.load_result()
    ss_300K.load_result()
    # x_270K, F_270K = ss_270K.x_u, ss_270K.F_nu_u
    x_300K, F_300K = ss_300K.x_u, ss_300K.F_nu_u
    x_300K = unp.nominal_values(x_300K)

    reweighted_F_only_bulk = reweight_free_energy(x_300K, F_300K, 300, 270)
    reweighted_F_all = reweight_free_energy(x_300K, F_300K, 300, 270, **info)

    title = "Comparison of Reweighting Methods"
    x_label = r"$x$"
    y_label = r"$F (kJ/mol)$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    # plot_with_error_bar(ax, x_270K, F_270K, label="270 K")
    plot_with_error_bar(ax, x_300K, F_300K, label="300 K")
    plot_with_error_bar(ax, x_300K, reweighted_F_only_bulk, label="Re_Mu")
    plot_with_error_bar(ax, x_300K, reweighted_F_all, label="Re_All")
    ax.legend()

    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/figure/comparision_reweighting_method.png"
    save_figure(fig, figure_save_dir)
    plt.close(fig)


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


def main():
    rho = 0.75
    process = "melting_300K"

    # post_processing_eda(rho, process)
    # post_processing_ss(rho, process)
    # post_processing_wham(rho, process)
    # post_processing_fit_spherical_cap(rho, process)
    # calc_plot_lambda_q(rho, process)
    # compare_reweighting_method()
    # compare_with_theory()
    # main_plot_R_theta(rho, process)
    main_Delta_T_star(rho, process)


if __name__ == "__main__":
    main()
