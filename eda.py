# Author: Mian Qin
# Date Created: 6/12/24
from pathlib import Path
import json

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import SpanSelector, Button
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import statsmodels.tsa.stattools
from uncertainties import unumpy as unp

from utils import calculate_histogram_parameters
from utils_plot import create_fig_ax, save_figure, plot_with_error_band
from op_dataset import OPDataset


class EDA:
    result_filename = "autocorr_time.json"

    def __init__(self, dataset: OPDataset, op, save_dir=Path("./figure")):
        self.dataset = dataset
        self.op = op
        self.save_dir = save_dir
        self.acf_dict: dict[str, unp.uarray] | None = None
        self.autocorr_time_dict: dict[str, list[float]] | None = None

    @property
    def tau_dict(self):
        tau_dict = {}
        for job_name, (tau_cross, tau_int, tau_fit) in self.autocorr_time_dict.items():
            tau = np.mean([tau_cross, tau_int, tau_fit])
            tau_dict[job_name] = tau
        return tau_dict

    def calculate(self):
        self.__calc_autocorr_func()
        # self._calc_autocorr_time()

    def __calc_autocorr_func(self):
        op = self.op
        acf_dict = {}
        for job_name, data in self.dataset.items():
            df = data.df
            values = df[op].values
            acf, confint = statsmodels.tsa.stattools.acf(values, nlags=len(df), fft=True, alpha=0.3173)  # Confidence interval: 1 sigma
            sigma = (confint[:, 1] - confint[:, 0]) / 2
            acf_u = unp.uarray(acf, sigma)
            acf_dict[job_name] = acf_u
        self.acf_dict = acf_dict

    def __determine_autocorr_time(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        job_name_list, acf_list = zip(*self.acf_dict.items())
        num_datasets = len(job_name_list)
        index = [0]

        def plot_data(index):
            job_name = job_name_list[index[0]]
            acf_u = acf_list[index[0]]
            t = self.dataset[job_name].df["t"].values
            t = t - t[0]

            plot_with_error_band(ax1, t, acf_u)
            ax1.set_ylabel('ACF')
            ax1.set_title('Autocorrelation Function')
            ax1.legend()
            ax1.grid(True)

            acf = unp.nominal_values(acf_u)
            acf_err = unp.std_devs(acf_u)
            with (np.errstate(divide='ignore', invalid='ignore')):
                log_acf = np.log(np.abs(acf))
                log_acf_err = acf_err / np.abs(acf)
                log_acf_u = unp.uarray(log_acf, log_acf_err)
            # Plot the log ACF
            plot_with_error_band(ax2, t, log_acf_u)
            ax2.set_xlabel('Lag time (τ)')
            ax2.set_ylabel('Log ACF')
            ax2.set_title('Log of Autocorrelation Function')
            ax2.legend()
            ax2.grid(True)

            fit_line = ax2.plot([], [], "r-")
            exp_fit_line = ax1.plot([], [], "r-")

            def onselect(xmin, xmax):
                # Extract the indices of the selected data range
                indmin, indmax = np.searchsorted(t, (xmin, xmax))
                indmax = min(len(t) - 1, indmax)
                if indmax - indmin < 5:
                    fig.canvas.draw_idle()
                    return

                # Extract selected data
                t_sel = t[indmin:indmax]
                log_acf_sel = log_acf[indmin:indmax]

                # Weigh the fit by the inverse of the variances
                p = np.polyfit(t_sel, log_acf_sel, 1)

                # Plot the fitted curve on ax1
                y_fit = np.polyval(p, t_sel)
                exp_y_fit = np.exp(y_fit)
                fit_line.set_data(t_sel, y_fit)
                exp_fit_line.set_data(t_sel, exp_y_fit)
                fig.canvas.draw_idle()

            span = SpanSelector(ax2, onselect, "horizontal", useblit=True)

            ax1.clear()
            ax2.clear()
            span.disconnect_events()

        plot_data(index)

        button_ax = plt.axes((0.8, 0.01, 0.1, 0.05))
        next_button = Button(button_ax, "Next")

        def on_next(event):
            index[0] += 1
            if index[0] >= num_datasets:
                plt.close(fig)  # 关闭图形窗口
                return
            plot_data(index)

        next_button.on_clicked(on_next)
        plt.show()

    def plot_op(self, save_fig=True, save_dir=Path("./figure")):
        figure_save_dir = save_dir / "autocorr_func_detail"
        op = self.op
        plt.style.use("presentation.mplstyle")
        for job_name, data in self.dataset.items():
            df = data.initial_df
            x = df["t"].values
            y = df[op].values
            y_smooth = gaussian_filter1d(y, sigma=10)

            fig, ax = plt.subplots()
            ax.set_title("Order Parameters as a Function of $t$")
            ax.set_xlabel("$t$(ps)")
            ax.set_ylabel(f"{op}")

            ax.plot(x, y, "b-")
            ax.plot(x, y_smooth, "r--", label="Smoothed")
            ax.legend()

            if save_fig:
                figure_save_dir.mkdir(parents=True, exist_ok=True)
                save_path = figure_save_dir / f"op_{op}_{job_name}.png"
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
                print(f"Saved the figure to {save_path.resolve()}")
            else:
                plt.show()
            plt.close(fig)

    def plot_autocorr(self, save_fig=True, save_dir=Path("./figure")):
        """
        Compute and plot the autocorrelation function for the given column.

        :param save_fig: Boolean indicating whether to save the figure to a file or display it.
        :param save_dir: Directory to save the figure.
        :return:
        """
        op = self.op
        plt.style.use("presentation.mplstyle")
        autocorr_func_dict = self.acf_dict
        # autocorr_time_dict = self.autocorr_time_dict
        colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(autocorr_func_dict)))

        # Plot all ACF in one plot
        fig, ax = plt.subplots()
        ax.set_title(f"Auto-correlation Function (ACF) of {op}")
        ax.set_xlabel("$t$(ps)")
        ax.set_ylabel("ACF")

        lines = []
        for job_name, color in zip(autocorr_func_dict, colors):
            autocorr_func = autocorr_func_dict[job_name]
            t = self.dataset[job_name].df["t"].values
            t = t[:len(autocorr_func)]
            t = t - t[0]
            line = ax.plot(t, autocorr_func, "-", color=color, label=job_name)
            lines.append(line[0])

        # lines = [lines[0], lines[-1]]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels)
        if save_fig:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"autocorr_func_{op}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()
        plt.close(fig)

    def plot_histogram(self, num_bins: int = None, bin_width: float = None,
                       bin_range: tuple[float, float] = None, save_fig=True, save_dir=Path("./figure")):
        """
        Generate and plot histograms for the 'x' column of each dataset in self.data.

        :param num_bins: Number of bins.
                         If not specified, calculates the number of bins using interval and bin_range.
        :param bin_width: The bin width for the histogram. Priority < num_bins.
                         If not specified, calculates the bin width using num_bins and bin_range
        :param bin_range: A tuple specifying the minimum and maximum range of the bins.
                          If not specified, uses the minimum and maximum range in the data.
        :param save_fig: Boolean indicating whether to save the figure to a file or display it.
        :param save_dir: Directory to save the figure.
        """
        op = self.op
        plt.style.use("presentation.mplstyle")
        num_bins, bin_range = calculate_histogram_parameters(self.dataset, op, num_bins, bin_width, bin_range)

        fig, ax = plt.subplots()
        for _, data in self.dataset.items():
            df = data.df
            hist, bin_edges = np.histogram(df[op], bins=num_bins, range=bin_range)
            bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

            valid_indices = hist >= 1
            hist_non_zero = hist[valid_indices]
            negative_log_hist = -np.log(hist_non_zero)
            bin_midpoints_non_zero = bin_midpoints[valid_indices]

            ax.plot(bin_midpoints_non_zero, negative_log_hist, marker="o")

        ax.set_title(f"Negative Log Histogram for {op}")
        ax.set_xlabel(f"{op}")
        ax.set_ylabel(r"$-\ln W$")
        if save_fig:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"NLhist_{op}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()
        plt.close(fig)

    def save_result(self, save_dir=Path("./figure")):
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / self.result_filename
        with open(save_path, 'w') as file:
            json.dump(self.autocorr_time_dict, file, indent="\t")
        print(f"Saved the result to {save_path.resolve()}")

    def load_result(self, save_dir=Path("./figure")):
        save_path = save_dir / self.result_filename
        with open(save_path, 'r') as file:
            self.autocorr_time_dict = json.load(file)


def find_best_line(x_array: np.ndarray, y_array: np.ndarray):
    min_length = int(len(y_array) / 10)
    candidate_list = []
    max_start = min(int(len(y_array) / 20 + 1), 10)
    for i_start in range(max_start):
        not_line_counter = 0
        for i_end in range(i_start + min_length, len(x_array)):
            index = np.arange(i_start, i_end + 1)
            x = x_array[index]
            y = y_array[index]
            k_connect = (y[-1] - y[0]) / (x[-1] - x[0])
            b_connect = y[0] - k_connect * x[0]
            y_connect = k_connect * x + b_connect
            above = y > y_connect
            if not (0.1 < np.mean(above) < 0.9):  # not a line
                not_line_counter += 1
                if not_line_counter >= 3:
                    break
                continue
            else:
                not_line_counter = 0

            p = np.polyfit(x, y, 1)
            if p[0] >= 0:
                continue
            y_fit = np.polyval(p, x)

            y_diff = y - y_fit
            mean_RSS = np.mean(y_diff ** 2)
            length = i_end - i_start

            candidate_list.append([mean_RSS, length, p, index])
    candidate_list.sort(key=lambda x: np.log(x[0]) - 1.5 * np.log(x[1]))
    mean_RSS, length, p, index = candidate_list[0]
    return p, index


def main():
    ...


if __name__ == "__main__":
    main()
