# Author: Mian Qin
# Date Created: 6/5/24
from pathlib import Path
import json

import numpy as np
import scipy.ndimage
from scipy.interpolate import CubicSpline

from utils import convert_unit
from op_dataset import OPDataset, load_dataset
from eda import EDA
from free_energy.wham import BinlessWHAM
from free_energy.sparse_sampling import SparseSampling

DATA_DIR = Path("/home/qinmian/data/gromacs/ice_melting/prd/melting/result")
FIGURE_SAVE_DIR = DATA_DIR / "figure"


def read_data() -> OPDataset:
    with open(DATA_DIR / "job_params.json", 'r') as file:
        job_params = json.load(file)

    dataset: OPDataset = load_dataset(
        data_dir=DATA_DIR,
        job_params=job_params,
        column_names=["t", "QBAR", "box.N", "box.Ntilde"],
        column_types={"t": float, "QBAR": float, "box.N": int,
                      "box.Ntilde": float},
    )
    dataset.drop_before(2000)
    return dataset


def plot_gradient():
    import matplotlib.pyplot as plt
    dataset = read_data()
    op = "QBAR"
    wham = BinlessWHAM(dataset, op)
    wham.load_result(save_dir=FIGURE_SAVE_DIR)
    x, y = wham.energy
    y = scipy.ndimage.gaussian_filter1d(y, sigma=1)
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)

    plt.style.use("presentation.mplstyle")
    fig, ax1 = plt.subplots()
    ax1.set_title("Gradient of Energy, WHAM")
    ax1.set_xlabel(f"{op}")
    ax1.set_ylabel(r"$\beta dF / dx$", color="blue")
    ax1.tick_params(axis='y', colors="blue")
    ax1.plot(x, convert_unit(dy), "b-")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$\beta d^2F / dx^2$", color="red")
    ax2.tick_params(axis='y', colors="red")
    ax2.plot(x, convert_unit(d2y), "r-")

    save_dir = DATA_DIR / "figure"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"gradient_energy.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")


def compare():
    import matplotlib.pyplot as plt
    dataset = read_data()
    op = "QBAR"

    wham = BinlessWHAM(dataset, op)
    wham.load_result(save_dir=FIGURE_SAVE_DIR)
    x1, y1 = wham.energy
    ss = SparseSampling(dataset, op)
    ss.load_result(save_dir=FIGURE_SAVE_DIR)
    x2, y2 = ss.energy
    y2 = y2 - 1.2 * (x2 > 206)  # An estimation of the integration of changing kappa

    plt.style.use("presentation.mplstyle")
    plt.figure()
    plt.plot(x1, convert_unit(y1), "r-", label="WHAM")
    plt.plot(x2, convert_unit(y2), "bo-", label="Sparse Sampling")
    plt.title("Box of Ice, WHAM vs SparseSampling")
    plt.xlabel(f"{op}")
    plt.ylabel(r"$\beta F$")
    plt.legend()

    save_dir = DATA_DIR / "figure"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"compare.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")

    x = np.linspace(min(x1.max(), x2.max()), max(x1.min(), x2.min()), 1000, endpoint=True)
    f_y1 = CubicSpline(x1, y1)
    f_y2 = CubicSpline(x2, y2)
    delta_y = f_y1(x) - f_y2(x)
    delta_y = scipy.ndimage.gaussian_filter1d(delta_y, sigma=1)
    delta_dy = np.gradient(delta_y, x)

    plt.style.use("presentation.mplstyle")
    fig, ax1 = plt.subplots()
    ax1.set_title(r"Gradient of Energy Difference, $\Delta F = F_{WHAM} - F_{SS}$")
    ax1.set_xlabel(f"{op}")
    ax1.set_ylabel(r"$\beta\Delta F$", color="red")
    ax1.tick_params(axis='y', colors="red")
    line1 = ax1.plot(x, convert_unit(delta_y), "r-", label=r"$\Delta F$")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$\beta d\Delta F / dx$", color="blue")
    ax2.tick_params(axis='y', colors="blue")
    line2 = ax2.plot(x, convert_unit(delta_dy), "b-", label=r"$d \Delta F / d x$")

    lines = [line1[0], line2[0]]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)

    save_dir = DATA_DIR / "figure"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"gradient_energy_difference.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {save_path.resolve()}")


def main():
    dataset = read_data()
    op = "QBAR"
    eda = EDA(dataset)
    eda.plot_autocorr(column_name=op, save_dir=FIGURE_SAVE_DIR)
    eda.plot_histogram(column_name=op, bin_range=(0, 432), save_dir=FIGURE_SAVE_DIR)
    wham = BinlessWHAM(dataset, op)
    wham.calculate([op])
    wham.plot(save_dir=FIGURE_SAVE_DIR)
    wham.save_result(save_dir=FIGURE_SAVE_DIR)
    ss = SparseSampling(dataset, op)
    ss.calculate()
    ss.plot(save_dir=FIGURE_SAVE_DIR, delta_mu=0.0)
    ss.plot_detail(save_dir=FIGURE_SAVE_DIR)
    ss.save_result(save_dir=FIGURE_SAVE_DIR)


if __name__ == "__main__":
    main()
    plot_gradient()
    compare()
