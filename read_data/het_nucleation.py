# Author: Mian Qin
# Date Created: 2025/1/15
from pathlib import Path
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

from utils import calculate_triangle_area
from utils_plot import create_fig_ax, save_figure, plot_with_error_bar
from op_dataset import OPDataset, load_dataset
from eda import EDA
from free_energy import SparseSampling
from free_energy_reweighting import reweight_free_energy

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
    ss.plot_different_DeltaT(save_dir=figure_save_dir)
    ss.plot_detail(save_dir=figure_save_dir)
    ss.save_result()


def calc_plot_lambda_q(rho, process):
    figure_save_dir = home_path / f"data/gromacs/het_nucleation/data/{rho}/prd/{process}/figure"
    dataset = read_data(rho, process)
    intermediate_result_save_dir = dataset.save_dir / "lambda_qbar"

    qbar_list = []
    lambda_chillplus_list = []
    # lambda_with_PI_list = []
    for job_name, op_data in dataset.items():
        df = op_data.df_prd
        qbar = df["QBAR"].values
        # lambda_with_PI = df["lambda_with_PI"].values
        lambda_chillplus = df["lambda_chillplus"].values
        qbar_list.append(qbar)
        lambda_chillplus_list.append(lambda_chillplus)
        # lambda_with_PI_list.append(lambda_with_PI)

    qbar_array = np.concatenate(qbar_list)
    lambda_chillplus_array = np.concatenate(lambda_chillplus_list)
    # lambda_with_PI_array = np.concatenate(lambda_with_PI_list)

    title = rf"Number of Ice-like Water, $\rho = {rho}$, {process}"
    x_label = "qbar"
    y_label = r"$\lambda$"
    fig, ax = create_fig_ax(title, x_label, y_label)
    ax.plot(qbar_array, lambda_chillplus_array, "b+", label="lambda_chillplus")
    # ax.plot(qbar_array, lambda_with_PI_array, 'g+', label="lambda_with_PI")

    ax.legend()
    save_path = figure_save_dir / "lambda_qbar.png"
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


def main():
    rho = 0.75
    process = "melting_270K"

    # post_processing_eda(rho, process)
    # post_processing_ss(rho, process)
    # calc_plot_lambda_q(rho, process)
    compare_reweighting_method()


if __name__ == "__main__":
    main()
