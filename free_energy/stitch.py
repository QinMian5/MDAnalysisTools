# Author: Mian Qin
# Date Created: 6/12/24
from itertools import pairwise
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import scipy.constants as c

from utils import calculate_histogram_parameters, convert_unit
from op_dataset import OPDataset


class Stitch:
    def __init__(self, dataset: OPDataset):
        self.dataset = dataset
        self.stitched_data = {}

    def stitch(self, column_name, num_bins: int = None,
               bin_width: float = None, bin_range: tuple[float, float] = None):
        num_bins, bin_range = calculate_histogram_parameters(self.dataset, column_name, num_bins, bin_width, bin_range)
        unbiased_data = []
        for job_name, data in self.dataset.items():
            df = data.df_prd
            params = data.params
            hist, bin_edges = np.histogram(df[column_name].to_numpy(), bins=num_bins, range=bin_range)
            bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

            valid_indices = hist >= 1
            hist_valid = hist[valid_indices]
            bin_midpoints_valid = bin_midpoints[valid_indices]

            bias_potential = data.calculate_bias_potential({column_name: bin_midpoints_valid})

            pl = hist_valid / np.sum(hist_valid)
            energy = -1 / data.beta * np.log(pl) / 1000 * c.N_A
            unbiased_free_energy = energy - bias_potential

            unbiased_data.append([job_name, params, bin_midpoints_valid, unbiased_free_energy])

        unbiased_data.sort(key=lambda x: np.mean(x[2]))

        offsets = [0]
        for data1, data2 in pairwise(unbiased_data):
            x1, y1 = data1[2:]
            x2, y2 = data2[2:]

            # Find the overlapping area
            left = max(x1[0], x2[0])
            right = min(x1[-1], x2[-2])

            x = np.linspace(left, right, 10, endpoint=True)
            interp_func1 = interp1d(x1, y1, kind="linear")
            interp_func2 = interp1d(x2, y2, kind="linear")

            y_interp1 = interp_func1(x)
            y_interp2 = interp_func2(x)
            mean_diff = np.mean(y_interp2 - y_interp1)
            offsets.append(offsets[-1] - mean_diff)

        stitched_data = []
        for offset, [job_name, bias_params, bin_midpoints_valid, unbiased_free_energy] in zip(offsets, unbiased_data):
            stitched_data.append([job_name, bias_params, bin_midpoints_valid, unbiased_free_energy + offset])

        self.stitched_data[column_name] = stitched_data

    def plot(self, column_name: str, save_fig=True, save_dir=Path("./figure")):
        plt.style.use("../presentation.mplstyle")
        plt.figure()
        for job_name, params, bin_midpoints_valid, unbiased_free_energy in self.stitched_data[column_name]:
            plt.plot(bin_midpoints_valid, convert_unit(unbiased_free_energy), label=params)

        plt.title(f"Stitch Plot for {column_name}")
        plt.xlabel(f"{column_name}")
        plt.ylabel(r"$\beta F$")
        if save_fig:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"Stitch_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()


def main():
    ...


if __name__ == "__main__":
    main()

