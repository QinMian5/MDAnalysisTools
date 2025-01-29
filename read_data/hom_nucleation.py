# Author: Mian Qin
# Date Created: 2025/1/15
from pathlib import Path
import json
import os
import pickle

import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt

from op_dataset import OPDataset, load_dataset
from eda import EDA
from free_energy import SparseSampling
from utils import calculate_triangle_area
from utils_plot import create_fig_ax, plot_with_error_bar, save_figure
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


def _load_params(process) -> dict:
    data_dir = home_path / f"data/gromacs/hom_nucleation/prd/{process}"
    with open(data_dir / "result" / "job_params.json", 'r') as file:
        job_params = json.load(file)
    return job_params


def read_data(process) -> OPDataset:
    data_dir = home_path / f"data/gromacs/hom_nucleation/prd/{process}/result"
    job_params = _load_params(process)

    dataset: OPDataset = load_dataset(
        data_dir=data_dir,
        job_params=job_params,
        file_type="csv",
        column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value", "lambda_chillplus"],
        column_types={"t": float, "QBAR": float, "box.N": int, "box.Ntilde": float, "bias_qbar.value": float,
                      "lambda_chillplus": int},
    )
    # dataset: OPDataset = load_dataset(
    #     data_dir=data_dir,
    #     job_params=job_params,
    #     file_type="out",
    #     column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value"],
    #     column_types={"t": float, "QBAR": float, "box.N": int, "box.Ntilde": float, "bias_qbar.value": float},
    # )
    return dataset


def post_processing_eda(process):
    figure_save_dir = home_path / f"data/gromacs/hom_nucleation/prd/{process}/figure"
    op = "QBAR"
    dataset = read_data(process)

    eda = EDA(dataset, op, figure_save_dir)
    eda.plot_ddF(figure_save_dir)
    eda.determine_relaxation_time()
    eda.calculate_acf()
    eda.determine_autocorr_time(figure_save_dir, ignore_previous=0)
    eda.plot_op(save_dir=figure_save_dir)
    eda.plot_histogram(bin_width=5, bin_range=(0, 1000), save_dir=figure_save_dir)
    eda.plot_acf(save_dir=figure_save_dir)
    eda.plot_act(save_dir=figure_save_dir)


def post_processing_ss(process):
    figure_save_dir = home_path / f"data/gromacs/hom_nucleation/prd/{process}/figure"
    op = "QBAR"
    dataset = read_data(process)

    ss = SparseSampling(dataset, op)
    ss.calculate()
    ss.plot_free_energy(save_dir=figure_save_dir)
    ss.plot_different_DeltaT(save_dir=figure_save_dir)
    ss.plot_detail(save_dir=figure_save_dir)
    ss.save_result()


def compare_reweighting_method():
    process_300K = "melting_300K"
    figure_save_dir = home_path / f"data/gromacs/hom_nucleation/prd/{process_300K}/figure"
    op = "QBAR"
    method_list = ["only_delta_mu", "all"]

    job_params_300K = _load_params(process_300K)
    # dataset_270K = read_data(rho, process_270K)
    dataset_300K = read_data(process_300K)
    interface_type_dict = {0: "IW"}
    info = {f"A_{v}": [] for v in interface_type_dict.values()}

    for job_name, params in job_params_300K.items():
        data_dir = dataset_300K.data_dir / job_name
        with open(data_dir / "interface.pickle", "rb") as file:
            # nodes, faces, interface_type = pickle.load(file)
            nodes, faces = pickle.load(file)
        _, A = calculate_triangle_area(nodes, faces)  # Unit: Angstrom^2
        A = A * 1e-20  # to unit m^2
        info[f"A_IW"].append(A)
    for k, v in info.items():
        info[k] = np.array(v)

    ss_300K = SparseSampling(dataset_300K, op)
    ss_300K.load_result()
    x_300K, F_300K = ss_300K.x_u, ss_300K.F_nu_u
    x_300K = unp.nominal_values(x_300K)

    T_m = 271

    for method in method_list:
        if method == "only_delta_mu":
            title = fr"Free Energy Profile at Different $\Delta T$, only $\Delta\mu$"
            x_label = fr"$x$"
            y_label = fr"$G(x;\Delta T)$ (kJ/mol)"
            fig, ax = create_fig_ax(title, x_label, y_label)

            for Delta_T in range(0, 121, 10):
                T = T_m - Delta_T
                label = fr"${Delta_T}\ \mathrm{{K}}$"
                reweighted_F_only_bulk = reweight_free_energy(x_300K, F_300K, 300, T)
                plot_with_error_bar(ax, x_300K, reweighted_F_only_bulk, label=label)

            ax.legend()
            save_path = figure_save_dir / f"sparse_sampling_DeltaT_only_delta_mu.png"
            save_figure(fig, save_path)
            plt.close(fig)
        elif method == "all":
            title = fr"Free Energy Profile at Different $\Delta T$, all"
            x_label = fr"$x$"
            y_label = fr"$G(x;\Delta T)$ (kJ/mol)"
            fig, ax = create_fig_ax(title, x_label, y_label)

            for Delta_T in range(0, 71, 10):
                T = T_m - Delta_T
                label = fr"${Delta_T}\ \mathrm{{K}}$"
                reweighted_F_all = reweight_free_energy(x_300K, F_300K, 300, T, **info)
                plot_with_error_bar(ax, x_300K, reweighted_F_all, label=label)

            ax.legend()
            save_path = figure_save_dir / f"sparse_sampling_DeltaT_all.png"
            save_figure(fig, save_path)
            plt.close(fig)


def main():
    process = "melting_300K"
    # post_processing_eda()
    # post_processing_ss(process)
    compare_reweighting_method()


if __name__ == "__main__":
    main()
