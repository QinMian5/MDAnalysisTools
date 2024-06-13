# Author: Mian Qin
# Date Created: 6/5/24
from pathlib import Path
import json

from utils import load_dataset, OPDataset
from eda import EDA
from sparse_sampling import SparseSampling

DATA_DIR = Path("/home/qinmian/data/gromacs/pseudoice/1.0/prd/melting/result")


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


def test_read_data():
    dataset = read_data()
    print()
    print(dataset["op_100"].df)


def main():
    figure_save_dir = DATA_DIR / "figure"
    dataset = read_data()
    eda = EDA(dataset)
    eda.plot_acf(column_name="QBAR", nlags=50, save_dir=figure_save_dir)
    eda.plot_histogram(column_name="QBAR", bin_width=2, bin_range=(0, 2000), save_dir=figure_save_dir)
    ss = SparseSampling(dataset, "QBAR")
    ss.calculate()
    ss.plot(save_dir=figure_save_dir)
    ss.plot_debug(save_dir=figure_save_dir)


if __name__ == "__main__":
    main()
