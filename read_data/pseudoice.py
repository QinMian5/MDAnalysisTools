# Author: Mian Qin
# Date Created: 6/5/24
from pathlib import Path
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import load_dataset, OPDataset, convert_unit
from eda import EDA
from sparse_sampling import SparseSampling

RHO = 1.0
DATA_DIR = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{RHO}/prd/melting/result")
FIGURE_SAVE_DIR = DATA_DIR / "figure"


def read_data() -> OPDataset:
    with open(DATA_DIR / "job_params.json", 'r') as file:
        job_params = json.load(file)

    dataset: OPDataset = load_dataset(
        data_dir=DATA_DIR,
        job_params=job_params,
        column_names=["t", "QBAR", "box.N", "box.Ntilde", "bias_qbar.value"],
        column_types={"t": float, "QBAR": float, "box.N": int,
                      "box.Ntilde": float, "bias_qbar.value": float},
    )
    dataset.drop_before(2000)
    return dataset


def plot_curvature_th(rho):
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    fig, ax = plt.subplots()
    save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/result/figure")
    ss = SparseSampling(None, op)
    ss.load_result(save_dir=save_dir)
    x, dF_lambda_dx = ss.dF_lambda_dx
    ax.plot(x, convert_unit(dF_lambda_dx), "o-", color="black")
    y = np.ones_like(x) * 0.24
    ax.plot(x, y, "--", color="black", label="Zero curvature")
    ax.set_title(rf"Mean Curvature, rho = {rho}")
    ax.set_xlabel(r"Number of ice-like molecules ($\lambda^*$)")
    ax.set_ylabel(r"$\beta \frac{dG}{d\lambda}$")
    ax.legend()

    save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/result/figure")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "curvature.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")


def plot_all():
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    rho_list = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    delta_mu = -0.22
    fig, ax = plt.subplots()
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(rho_list)))
    for rho, color in zip(rho_list, colors):
        save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/result/figure")
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


def plot_all_debug():
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    rho_list = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    fig, ax = plt.subplots()
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(rho_list)))
    for rho, color in zip(rho_list, colors):
        save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/result/figure")
        ss = SparseSampling(None, op)
        ss.load_result(save_dir=save_dir)
        x_star, dF_lambda_dx_star = ss.dF_lambda_dx
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


def main():
    dataset = read_data()
    eda = EDA(dataset)
    eda.plot_acf(column_name="QBAR", nlags=50, save_dir=FIGURE_SAVE_DIR)
    eda.plot_histogram(column_name="QBAR", bin_width=2, bin_range=(0, 2000), save_dir=FIGURE_SAVE_DIR)
    ss = SparseSampling(dataset, "QBAR")
    ss.calculate()
    ss.plot(save_dir=FIGURE_SAVE_DIR, delta_mu=-0.22)
    ss.plot_debug(save_dir=FIGURE_SAVE_DIR)
    ss.save_result(save_dir=FIGURE_SAVE_DIR)


if __name__ == "__main__":
    # main()
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        plot_curvature_th(rho)
    # plot_all()
    # plot_all_debug()
