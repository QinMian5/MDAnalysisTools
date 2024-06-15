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

DATA_DIR = Path("/home/qinmian/data/gromacs/pseudoice/0.5/prd/melting/result")
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


def plot_all():
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    rho_list = ["0.0", "0.5", "1.0"]
    k = -0.22
    fig, ax = plt.subplots()
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, 3))
    for rho, color in zip(rho_list, colors):
        save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/{rho}/prd/melting/result/figure")
        ss = SparseSampling(None, op)
        ss.load_result(save_dir=save_dir)
        x, energy = ss.energy
        ax.plot(x, convert_unit(energy) + k * x, "o-", color=color, label=rf"$\rho = {rho}$")
    ax.set_title(rf"Free Energy Profile Variation with Polarity $(\rho)$, $k = {k}$")
    ax.set_xlabel("Number of ice-like molecules")
    ax.set_ylabel(r"$\beta F + kN$")
    ax.legend()

    save_dir = Path("/home/qinmian/data/gromacs/pseudoice/figure")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "energy_profile.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")


def plot_all_debug():
    plt.style.use("presentation.mplstyle")

    op = "QBAR"
    rho_list = ["0.0", "0.5", "1.0"]
    fig, ax = plt.subplots()
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, 3))
    for rho, color in zip(rho_list, colors):
        save_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/{rho}/prd/melting/result/figure")
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
    ss.plot(save_dir=FIGURE_SAVE_DIR, k=-0.22)
    ss.plot_debug(save_dir=FIGURE_SAVE_DIR)
    ss.save_result(save_dir=FIGURE_SAVE_DIR)


if __name__ == "__main__":
    # main()
    plot_all()
    plot_all_debug()
