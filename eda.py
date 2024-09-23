# Author: Mian Qin
# Date Created: 6/12/24
from pathlib import Path
import json

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import acf

from utils import calculate_histogram_parameters
from op_dataset import OPDataset


class EDA:
    result_filename = "autocorr_time.json"

    def __init__(self, dataset: OPDataset, op, save_dir=Path("./figure")):
        self.dataset = dataset
        self.op = op
        self.save_dir = save_dir
        self.autocorr_func_dict: dict[str, np.ndarray]
        self.autocorr_time_dict: dict[str, list[float]]

    @property
    def tau_dict(self):
        tau_dict = {}
        for job_name, (tau_cross, tau_int, tau_fit) in self.autocorr_time_dict.items():
            tau = np.mean([tau_cross, tau_int, tau_fit])
            tau_dict[job_name] = tau
        return tau_dict

    def calculate(self):
        self._calc_autocorr_func()
        # self._calc_autocorr_time()

    def _calc_autocorr_func(self):
        op = self.op
        autocorr_dict = {}
        for job_name, data in self.dataset.items():
            df = data.df
            values = df[op].values
            autocorr_func = acf(values, nlags=len(df), fft=True)
            autocorr_func = autocorr_func
            autocorr_dict[job_name] = autocorr_func
        self.autocorr_func_dict = autocorr_dict

    def _calc_autocorr_time(self):
        figure_save_dir = self.save_dir / "autocorr_func_detail"
        op = self.op
        tau_dict = {}
        cross_threshold = 1 / np.e
        for job_name, autocorr_func in self.autocorr_func_dict.items():
            # Find the first intersection with the x-axis
            smoothed_acf = gaussian_filter1d(autocorr_func, sigma=5)
            for i in range(len(smoothed_acf) - 1):
                x1, x2 = smoothed_acf[i:i + 2]
                if x1 > cross_threshold > x2 or x1 < cross_threshold < x2:
                    tau_cross = i + np.abs(x1) / (np.abs(x1) + np.abs(x2))
                    break
            else:
                raise RuntimeError("ACF has no intersection with the x-axis")

            first_negative = np.where(autocorr_func <= 0)[0][0]
            tau_int: float = simpson(autocorr_func[:int(tau_cross / 0.618)])
            if first_negative < 10:
                tau_fit = tau_cross
            else:
                t = self.dataset[job_name].df["t"].values
                t = t - t[0]
                n_positive = int(first_negative)
                x = t[:n_positive]
                log_acf = np.log(autocorr_func[:n_positive])
                p, index = find_best_line(x, log_acf)
                x_match = x[index]
                y_match = log_acf[index]
                k, b = p
                x_line = np.array([x_match.min(), x_match.max()])
                y_line = k * x_line + b
                tau_fit = -1 / k
                plt.style.use("presentation.mplstyle")
                fig, ax = plt.subplots()
                ax.plot(x, log_acf, "b-")
                ax.plot(x_match, y_match, "ro", alpha=0.25)
                ax.plot(x_line, y_line, "r-", label=rf"$y={k:.4}x+{b:.4}$")
                ax.legend()
                ax.set_title(rf"Log of ACF of {op} for {job_name}, $\tau = {tau_fit:.2f}$ ps")
                # ax.set_title(rf"Log of ACF of {op} for {job_name}")
                ax.set_xlabel("$t$(ps)")
                ax.set_ylabel(r"$\log$ ACF")

                figure_save_dir.mkdir(parents=True, exist_ok=True)
                save_path = figure_save_dir / f"log_ACF_{op}_{job_name}.png"
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
                print(f"Saved the figure to {save_path.resolve()}")
                plt.close(fig)

                tau_fit /= self.dataset[job_name].time_step

            tau_dict[job_name] = [tau_cross, tau_int, tau_fit]
        self.autocorr_time_dict = tau_dict

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
        autocorr_func_dict = self.autocorr_func_dict
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

        lines = [lines[0], lines[-1]]
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

        # # Plot each ACF
        # detail_save_dir = save_dir / "autocorr_func_detail"
        # detail_save_dir.mkdir(parents=True, exist_ok=True)
        # for job_name in autocorr_func_dict:
        #     autocorr_func = autocorr_func_dict[job_name]
        #     autocorr_time = autocorr_time_dict[job_name]
        #     tau_cross, tau_int, tau_fit = autocorr_time
        #     tau = np.mean([tau_cross, tau_int, tau_fit])
        #     fig, ax = plt.subplots()
        #     ax.set_title(rf"ACF of {op} for {job_name}, $\bar\tau = {tau:.2f}$ ps")
        #     ax.set_xlabel("$t$(ps)")
        #     ax.set_ylabel("ACF")
        #     t = self.dataset[job_name].df["t"].values
        #     t = t[:len(autocorr_func)]
        #     t = t - t[0]
        #     index = slice(0, int(np.ceil(3 * tau)))
        #     x, y = t[index], autocorr_func[index]
        #     ax.plot([x.min(), x.max()], [0, 0], "--", color="black")
        #     ax.plot(x, y, "b-")
        #     ax.plot(tau_cross, 0, "ro", label=rf"$\tau_{{cross}} = {tau_cross:.2f}$ ps")
        #     ax.plot(tau_int, 0, "go", label=rf"$\tau_{{int}} = {tau_int:.2f}$ ps")
        #     ax.plot(tau_fit, 0, "bo", label=rf"$\tau_{{fit}} = {tau_fit:.2f}$ ps")
        #     ax.legend()
        #     if save_fig:
        #         save_path = detail_save_dir / f"autocorr_func_{op}_{job_name}.png"
        #         plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        #         print(f"Saved the figure to {save_path.resolve()}")
        #     else:
        #         plt.show()
        #     plt.close(fig)
        #
        # # Plot ACT
        # fig, ax = plt.subplots()
        # ax.set_title(f"Autocorrelation Time (ACT) of {op}")
        # # ax.set_xlabel("")
        # ax.set_ylabel("$t$(ps)")
        #
        # job_names = self.tau_dict.keys()
        # autocorr_times = self.tau_dict.values()
        #
        # ax.bar(job_names, autocorr_times)
        # plt.xticks(rotation=90)
        #
        # if save_fig:
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     save_path = save_dir / f"autocorr_time_{op}.png"
        #     plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        #     print(f"Saved the figure to {save_path.resolve()}")
        # else:
        #     plt.show()
        # plt.close(fig)

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


# def find_line(x_array: np.ndarray, y_array: np.ndarray) -> tuple[float, float, np.ndarray]:
#     N = x_array.shape[0]
#     assert x_array.shape == (N,) and y_array.shape == (N,), "Wrong shape"
#
#     # skip the first points
#     n_skip = 1
#     valid_index1 = (np.arange(n_skip, x_array.shape[0]),)
#     x_array = x_array[valid_index1]
#     y_array = y_array[valid_index1]
#
#     y_smooth_array = gaussian_filter1d(y_array, sigma=5)
#     fluctuation = np.abs(y_array - y_smooth_array)
#     smoothed_fluctuation = gaussian_filter1d(fluctuation, sigma=3)
#     threshold1 = 1.5 * smoothed_fluctuation[:int(0.1 * len(smoothed_fluctuation))].mean()  # The average of the first 10%
#     # threshold2 =
#     threshold = threshold1
#
#     valid_index2 = np.where(smoothed_fluctuation < threshold)
#     x_array = x_array[valid_index2]
#     y_array = y_array[valid_index2]
#
#     N = x_array.shape[0]
#     record_list = []
#     theta_array = np.linspace(0, np.pi / 2, 900)
#     for theta in theta_array:
#         r_array = x_array * np.cos(theta) + y_array * np.sin(theta)
#         sorted_r = np.sort(r_array)
#         bin_min = float(sorted_r[int(0.1 * N)])
#         bin_max = float(sorted_r[int(0.9 * N)])
#         bin_interval = threshold
#         bin_num = int((bin_max - bin_min) / bin_interval)
#         hist, bin_edges = np.histogram(r_array, bins=bin_num, range=(bin_min, bin_max))
#
#         # Gaussian Smoothing
#         sigma = 2
#         smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=sigma)
#         bin_center = (bin_edges[1:] + bin_edges[:-1]) / 2
#
#         indices = np.argsort(smoothed_hist)[::-1]
#         for index in indices[:5]:
#             r = bin_center[index]
#             n = hist[index]
#             if n < 5:
#                 continue
#             match_index = np.where(np.abs(r_array - r) < bin_interval)
#             real_index = valid_index1[0][valid_index2][match_index]
#             if real_index[0] > 10:
#                 continue
#             record_list.append((n, theta, r, real_index))
#     record_list.sort(key=lambda x: x[0], reverse=True)
#     theta, r, index = record_list[0][1:]
#     index = valid_index1[0][valid_index2]
#     return theta, r, index


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
