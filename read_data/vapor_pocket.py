# Author: Mian Qin
# Date Created: 6/12/24
from pathlib import Path
import json

from utils import load_dataset, convert_unit
from op_dataset import OPDataset
from eda import EDA
from stitch import Stitch
from wham import BinlessWHAM
from sparse_sampling import SparseSampling


def read_data() -> OPDataset:
    data_dir = Path("/home/qinmian/data/gromacs/vapor_pocket/prd/result")
    with open(data_dir / "job_params.json", 'r') as file:
        job_params = json.load(file)

    dataset: OPDataset = load_dataset(
        data_dir=data_dir,
        job_params=job_params,
        column_names=["t", "N", "NTILDE"],
        column_types={"t": float, "N": int, "NTILDE": float},
    )
    dataset.drop_before(100)
    return dataset


def compare():
    import matplotlib.pyplot as plt
    dataset = read_data()
    op = "NTILDE"
    wham = BinlessWHAM(dataset, op)
    wham.load_result(FIGURE_SAVE_DIR)
    ss = SparseSampling(dataset, op)
    ss.load_result(FIGURE_SAVE_DIR)

    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    x1, y1 = wham.energy
    x2, y2 = ss.energy
    ax.plot(x1, convert_unit(y1), "r-", label="WHAM")
    ax.plot(x2, convert_unit(y2), "bo-", label="Sparse Sampling")
    ax.set_title("Vapor Pocket, WHAM vs SparseSampling")
    ax.set_xlabel(f"{op}")
    ax.set_ylabel(r"$\beta F$")
    plt.legend()

    save_dir = FIGURE_SAVE_DIR
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"compare.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)


def main():
    figure_save_dir = Path("/home/qinmian/data/gromacs/vapor_pocket/prd") / "figure"
    op = "NTILDE"
    dataset = read_data()
    eda = EDA(dataset)
    eda.calculate(column_name=op)
    eda.plot_autocorr(column_name=op, save_dir=figure_save_dir)
    eda.plot_histogram(column_name=op, save_dir=figure_save_dir)
    stitch = Stitch(dataset)
    stitch.stitch(column_name=op)
    stitch.plot(column_name=op, save_dir=figure_save_dir)
    wham = BinlessWHAM(dataset, op)
    wham.calculate([op])
    wham.plot(save_dir=figure_save_dir)
    wham.save_result(save_dir=figure_save_dir)
    ss = SparseSampling(dataset, op)
    ss.calculate()
    ss.plot(save_dir=figure_save_dir)
    ss.plot_debug(save_dir=figure_save_dir)
    ss.save_result(save_dir=figure_save_dir)


if __name__ == "__main__":
    main()
    compare()
