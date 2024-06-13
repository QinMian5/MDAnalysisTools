# Author: Mian Qin
# Date Created: 6/12/24
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf

from utils import OPDataset, calculate_histogram_parameters


class EDA:
    def __init__(self, dataset: OPDataset):
        """
        Initialize the DataAnalysis object.

        :param data: An instance of the UmbrellaSamplingData that stores the simulation data.
        """
        self.dataset = dataset

    def plot_acf(self, column_name, nlags=None, save_fig=True, save_dir=Path("./figure")):
        """
        Compute and plot the autocorrelation function for the given column.

        :param column_name: Name of the column to compute the autocorrelation for.
        :param nlags: Number of lags to compute the autocorrelation for.
                      Default is min(10 * np.log10(nobs), nobs - 1).
        :param save_fig: Boolean indicating whether to save the figure to a file or display it.
        :param save_dir: Directory to save the figure.
        :return:
        """
        plt.style.use("presentation.mplstyle")
        plt.figure()
        for job_name, data in self.dataset.items():
            df = data.df
            t = df["t"].values
            data_to_acf = df[column_name].values
            autocorr = acf(data_to_acf, nlags=nlags, fft=True)
            plt.plot(t[:len(autocorr)]-t[0], autocorr, "-", label=job_name)
        plt.title(f"Autocorrelation Function for {column_name}")
        plt.xlabel("$t$(ps)")
        plt.ylabel("ACF")
        if save_fig:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"acf_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()

    def plot_histogram(self, column_name, num_bins: int = None, bin_width: float = None,
                       bin_range: tuple[float, float] = None, save_fig=True, save_dir=Path("./figure")):
        """
        Generate and plot histograms for the 'x' column of each dataset in self.data.

        :param column_name: Name of the column to compute and plot the histogram for.
        :param num_bins: Number of bins.
                         If not specified, calculates the number of bins using interval and bin_range.
        :param bin_width: The bin width for the histogram. Priority < num_bins.
                         If not specified, calculates the bin width using num_bins and bin_range
        :param bin_range: A tuple specifying the minimum and maximum range of the bins.
                          If not specified, uses the minimum and maximum range in the data.
        :param save_fig: Boolean indicating whether to save the figure to a file or display it.
        :param save_dir: Directory to save the figure.
        """
        plt.style.use("presentation.mplstyle")
        num_bins, bin_range = calculate_histogram_parameters(self.dataset, column_name, num_bins, bin_width, bin_range)

        plt.figure()
        for _, data in self.dataset.items():
            df = data.df
            hist, bin_edges = np.histogram(df[column_name], bins=num_bins, range=bin_range)
            bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

            valid_indices = hist >= 1
            hist_non_zero = hist[valid_indices]
            negative_log_hist = -np.log(hist_non_zero)
            bin_midpoints_non_zero = bin_midpoints[valid_indices]

            plt.plot(bin_midpoints_non_zero, negative_log_hist, marker="o")

        plt.title(f"Negative Log Histogram for {column_name}")
        plt.xlabel(f"{column_name}")
        plt.ylabel(r"$-\ln W$")
        if save_fig:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"NLhist_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()


def main():
    ...


if __name__ == "__main__":
    main()
