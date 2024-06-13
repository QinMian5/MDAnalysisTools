# Author: Mian Qin
# Date Created: 6/12/24
from pathlib import Path
import json

from utils import load_dataset, OPDataset
from eda import EDA
from stitch import Stitch
from wham import BinlessWHAM
from sparse_sampling import SparseSampling

DATA_DIR = Path("/home/qinmian/data/gromacs/vapor_pocket/prd/result")
FIGURE_SAVE_DIR = DATA_DIR / "figure"


def read_data() -> OPDataset:
    with open(DATA_DIR / "job_params.json", 'r') as file:
        job_params = json.load(file)

    dataset: OPDataset = load_dataset(
        data_dir=DATA_DIR,
        job_params=job_params,
        column_names=["t", "N", "NTILDE"],
        column_types={"t": float, "N": int, "NTILDE": float},
    )
    dataset.drop_before(100)
    return dataset


def compare():
    import matplotlib.pyplot as plt
    dataset = read_data()
    wham = BinlessWHAM(dataset, "NTILDE")
    wham.load_result(FIGURE_SAVE_DIR)
    ss = SparseSampling(dataset, "NTILDE")
    ss.load_result(FIGURE_SAVE_DIR)

    plt.style.use("presentation.mplstyle")
    plt.figure()
    x1, y1 = wham.energy
    x2, y2 = ss.F_nu
    y2 = y2 + 145
    plt.plot(x1, y1, "r-", label="WHAM")
    plt.plot(x2, y2, "bo-", label="Sparse Sampling")
    plt.legend()

    save_dir = FIGURE_SAVE_DIR
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"compare.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)


def main():
    op = "NTILDE"
    dataset = read_data()
    eda = EDA(dataset)
    eda.plot_acf(column_name=op, nlags=50, save_dir=FIGURE_SAVE_DIR)
    eda.plot_histogram(column_name=op, save_dir=FIGURE_SAVE_DIR)
    stitch = Stitch(dataset)
    stitch.stitch(column_name=op)
    stitch.plot(column_name=op, save_dir=FIGURE_SAVE_DIR)
    wham = BinlessWHAM(dataset, op)
    wham.calculate([op])
    wham.plot(save_dir=FIGURE_SAVE_DIR)
    wham.save_result(save_dir=FIGURE_SAVE_DIR)
    ss = SparseSampling(dataset, op)
    ss.calculate()
    ss.plot(save_dir=FIGURE_SAVE_DIR)
    ss.save_result(save_dir=FIGURE_SAVE_DIR)


if __name__ == "__main__":
    # main()
    compare()
