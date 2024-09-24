# Author: Mian Qin
# Date Created: 6/5/24
from pathlib import Path
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from utils import load_dataset, convert_unit, read_solid_like_atoms
from utils_plot import create_fig_ax, save_figure
from op_dataset import OPDataset
from eda import EDA
from sparse_sampling import SparseSampling

_filename_job_params = "job_params.json"
_filename_index = "solid_like_atoms.index"
_filename_index2 = "solid_like_atoms_with_PI.index"
_filename_index3 = "solid_like_atoms_chillplus.index"
_filename_index4 = "solid_like_atoms_corrected.index"
_filename_lambda_q = "lambda_q.json"


def _load_params(rho, process) -> dict:
    data_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/{process}/result")
    with open(data_dir / _filename_job_params, 'r') as file:
        job_params = json.load(file)
    return job_params


def read_data(rho, process) -> OPDataset:
    data_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/{process}/result")
    job_params = _load_params(rho, process)

    dataset: OPDataset = load_dataset(
        data_dir=data_dir,
        job_params=job_params,
        column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value"],
        column_types={"t": float, "QBAR": float, "box.N": int,
                      "box.Ntilde": float, "bias_qbar.value": float},
    )
    return dataset


def filter_solid_like_atoms(solid_like_atoms_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    for k, v, in solid_like_atoms_dict.items():
        for i in range(len(v)):
            if int(v[i]) > 11892:
                solid_like_atoms_dict[k] = v[:i]
                break
    return solid_like_atoms_dict


def get_qbar_from_dataset(dataset: OPDataset) -> dict[str, pd.DataFrame]:
    q_dict = {}
    for job_name, data in dataset.items():
        df = data.df
        t = [f"{x:.1f}" for x in df["t"].values]
        qbar = df["QBAR"].values
        q_dict[job_name] = pd.DataFrame({"t": t, "qbar": qbar})
    return q_dict


def read_lambda(rho, process, filename=_filename_index, column_name="lambda") -> dict[str, pd.DataFrame]:
    data_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/{process}/result")
    job_params = _load_params(rho, process)

    lambda_dict: dict[str, pd.DataFrame] = {}
    for job_name in job_params:
        filepath = data_dir / job_name / filename
        solid_like_atoms_dict = filter_solid_like_atoms(read_solid_like_atoms(filepath))
        t = list(solid_like_atoms_dict.keys())
        lambda_ = [len(indices) for indices in solid_like_atoms_dict.values()]
        lambda_dict[job_name] = pd.DataFrame({"t": t, column_name: lambda_})
    return lambda_dict


def plot_all_in_one():
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    rho_list = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    delta_mu = -0.22
    fig, ax = plt.subplots()
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(rho_list)))
    for rho, color in zip(rho_list, colors):
        save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/figure")
        ss = SparseSampling(None, op)
        ss.load_result(save_dir=save_dir)
        x, energy = ss.energy
        ax.plot(x, convert_unit(energy) + delta_mu * x, "o-", color=color, label=rf"$\rho = {rho}$")
    ax.set_title(rf"Free Energy Profile Variation with Polarity $(\rho)$, $\Delta\mu = {delta_mu} k_BT$")
    ax.set_xlabel("Number of ice-like molecules")
    ax.set_ylabel(r"$\beta F + \Delta\mu N$")
    ax.legend()

    save_dir = Path("/home/qinmian/data/gromacs/pseudoice/figure")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "energy_profile.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")


def plot_all_debug_in_one():
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    rho_list = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    fig, ax = plt.subplots()
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(rho_list)))
    for rho, color in zip(rho_list, colors):
        save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/result/figure")
        ss = SparseSampling(None, op)
        ss.load_result(save_dir=save_dir)
        x_star, dF_lambda_dx_star = ss.dF_lambda_dx_star
        ax.plot(x_star, convert_unit(dF_lambda_dx_star), "o-", color=color, label=rf"$\rho = {rho}$")
    ax.set_title(rf"Sparse Sampling (Debug)")
    ax.set_xlabel("$N^*$")
    ax.set_ylabel(r"$\beta \frac{dF_{\lambda}}{d{N^*}}$")
    ax.legend()

    save_dir = Path("/home/qinmian/data/gromacs/pseudoice/figure")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "sparse_sampling_debug.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")


def calc_plot_save(rho, process):
    figure_save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/{process}") / "figure"
    op = "QBAR"
    dataset = read_data(rho, process)
    eda = EDA(dataset, op, figure_save_dir)
    # eda.load_result(figure_save_dir)
    eda.plot_op(save_dir=figure_save_dir)
    eda.plot_histogram(bin_width=2, bin_range=(0, 1800), save_dir=figure_save_dir)
    eda.calculate()
    eda.plot_autocorr(save_dir=figure_save_dir)
    # eda.save_result(save_dir=figure_save_dir)
    # tau_dict = eda.tau_dict
    # dataset.update_autocorr_time(tau_dict)
    ss = SparseSampling(dataset, op)
    ss.calculate()
    ss.plot_free_energy(save_dir=figure_save_dir)
    ss.plot_different_DeltaT(save_dir=figure_save_dir)
    # ss.plot_debug(save_dir=figure_save_dir)
    ss.save_result(save_dir=figure_save_dir)


def calc_plot_lambda_q(rho, process):
    figure_save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/{process}") / "figure"
    dataset = read_data(rho, process)
    qbar_dict = get_qbar_from_dataset(dataset)
    q_dict = read_lambda(rho, _filename_index, column_name="q")
    lambda_dict = read_lambda(rho, _filename_index4, column_name="lambda")
    job_params = _load_params(rho, process)

    qbar_star_list = []
    qbar_avg_list = []
    q_avg_list = []
    lambda_avg_list = []
    for job_name in job_params:
        qbar_star = job_params[job_name]["QBAR"]["CENTER"]
        qbar_star_list.append(qbar_star)

        df = qbar_dict[job_name].merge(q_dict[job_name], on="t").merge(lambda_dict[job_name], on="t")
        qbar_array = df["qbar"].values
        qbar_avg = qbar_array.mean()
        qbar_avg_list.append(qbar_avg)
        q_array = df["q"].values
        q_avg = q_array.mean()
        q_avg_list.append(q_avg)
        lambda_array = df["lambda"].values
        lambda_avg = lambda_array.mean()
        lambda_avg_list.append(lambda_avg)

    qbar_star_array = np.array(qbar_star_list)
    qbar_avg_array = np.array(qbar_avg_list)
    q_avg_array = np.array(q_avg_list)
    lambda_avg_array = np.array(lambda_avg_list)

    # Plot q_avg, lambda_avg as a function of q_star
    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.plot(qbar_star_array, qbar_avg_array, "b-o", label="qbar")
    ax.plot(qbar_star_array, q_avg_array, 'g-o', label="q")
    ax.plot(qbar_star_array, lambda_avg_array, 'r-o', label="lambda")
    ax.legend()
    ax.set_title(rf"Number of Ice-like Molecules, $\rho = {rho}$")
    ax.set_xlabel("$q^*$")
    ax.set_ylabel("number of ice-like molecules")
    save_path = figure_save_dir / "number_of_ice.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")
    plt.close(fig)

    # Fit lambda_avg as a function of q_avg
    def linear(x, p0, p1):
        return p0 * x + p1

    p_opt, p_cov = curve_fit(linear, q_avg_array, lambda_avg_array)
    p_err = np.sqrt(np.diag(p_cov))
    r, _ = pearsonr(q_avg_array, lambda_avg_array)
    magnitude = -np.floor(np.log10(np.abs(p_err))).astype(int) + 1
    magnitude_r = -np.floor(np.log10(1 - r)).astype(int) + 1
    x = np.linspace(q_avg_array.min(), q_avg_array.max(), 100)
    y = linear(x, *p_opt)

    fig, ax = plt.subplots()
    ax.set_title(rf"$\rho = {rho}$")
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\lambda$")
    ax.plot(q_avg_array, lambda_avg_array, "bo")
    ax.plot(x, y, "r-")
    text = '\n'.join([rf"$\lambda = k q + b$",
                      rf"$k = {p_opt[0]:.{magnitude[0]}f} \pm {p_err[0]:.{magnitude[0]}f}$",
                      rf"$b = {p_opt[1]:.{magnitude[1]}f} \pm {p_err[1]:.{magnitude[1]}f}$",
                      rf"$r = {r:.{magnitude_r}f}$"])
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=14, verticalalignment='top')

    save_path = figure_save_dir / "lambda_q.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")
    plt.close(fig)

    save_path = figure_save_dir / _filename_lambda_q
    with open(save_path, 'w') as file:
        json.dump([p_opt[0], p_err[0]], file)
        print(f"Saved the fitting parameters to {save_path.resolve()}")


def plot_g_lambda(rho):
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    fig, ax = plt.subplots()
    save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/figure")
    ss = SparseSampling(None, op)
    ss.load_result(save_dir=save_dir)
    x = ss.x
    dG_dq, sigma_dG_dq = ss.dF_nu_dx
    with open(save_dir / _filename_lambda_q) as file:
        dlambda_dq, sigma_dlambda_dq = json.load(file)

    dG_dlambda = dG_dq / dlambda_dq
    sigma_dG_dlambda = np.abs(dG_dlambda) * np.sqrt((sigma_dG_dq / dG_dq) ** 2 + (sigma_dlambda_dq / dlambda_dq) ** 2)
    ax.errorbar(x, convert_unit(dG_dlambda), yerr=3*convert_unit(sigma_dG_dlambda), fmt='b-o', ecolor='red', capsize=5,
                capthick=1, elinewidth=1)
    y = np.ones_like(x) * 0.24
    ax.plot(x, y, "--", color="black", label="Zero curvature")
    ax.set_title(rf"$g_\lambda$, $\rho = {rho}$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\beta \frac{dG}{d\lambda}$")
    ax.legend()

    save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/figure")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "curvature.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")
    plt.close(fig)


def compare_melting_icing(rho):
    figure_save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice") / "figure"

    op = "QBAR"
    process_list = ["melting", "icing"]
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


def main():
    process = "icing_300_long_ramp"
    for rho in [0.75]:
        calc_plot_save(rho, process)
        # calc_plot_lambda_q(rho)
        # plot_g_lambda(rho)

        # compare_melting_icing(rho)


if __name__ == "__main__":
    main()
