# Author: Mian Qin
# Date Created: 6/12/24
from pathlib import Path
import json

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import SpanSelector, Button, TextBox
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
        self.act_dict: dict[str, float] | None = None
        self.autocorr_time_dict: dict[str, list[float]] | None = None

    @property
    def tau_dict(self):
        tau_dict = {}
        for job_name, (tau_cross, tau_int, tau_fit) in self.autocorr_time_dict.items():
            tau = np.mean([tau_cross, tau_int, tau_fit])
            tau_dict[job_name] = tau
        return tau_dict

    def calculate_acf(self):
        self.__calc_autocorr_func()

    def __calc_autocorr_func(self):
        op = self.op
        acf_dict = {}
        for job_name, data in self.dataset.items():
            df = data.df
            values = df[op].values
            acf, confint = statsmodels.tsa.stattools.acf(values, nlags=len(df), fft=True,
                                                         alpha=0.3173)  # Confidence interval: 1 sigma
            sigma = (confint[:, 1] - confint[:, 0]) / 2
            acf_u = unp.uarray(acf, sigma)
            acf_dict[job_name] = acf_u
        self.acf_dict = acf_dict

    def determine_autocorr_time(self, figure_save_dir=Path("./figure"), ignore_previous=True):
        acf_dict = {}
        for job_name, acf_u in self.acf_dict.items():
            if ignore_previous and self.dataset[job_name].autocorr_time is not None:
                continue
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            t = self.dataset[job_name].df["t"].values
            t = t - t[0]

            plot_with_error_band(ax1, t, acf_u)
            ax1.plot(t, np.zeros_like(t), "--", color="black")
            ax1.set_ylabel('ACF')
            ax1.set_title(f"Autocorrelation Function")

            acf = unp.nominal_values(acf_u)
            acf_err = unp.std_devs(acf_u)
            plot_until = np.where(acf < 1e-3)[0][0]
            log_acf = np.log(acf[:plot_until])
            log_acf_err = acf_err[:plot_until] / acf[:plot_until]
            log_acf_u = unp.uarray(log_acf, log_acf_err)
            # Plot the log ACF
            plot_with_error_band(ax2, t[:plot_until], log_acf_u)
            ax2.set_xlabel('Lag time (τ)')
            ax2.set_ylabel('Log ACF')
            ax2.set_title("Log of Autocorrelation Function")

            fit_line = ax2.plot([], [], "r-")[0]
            exp_fit_line = ax1.plot([], [], "r-")[0]
            act_line = ax1.plot([], [], "r--")[0]

            text_box_ax = fig.add_axes((0.2, 0.01, 0.1, 0.05))
            text_box = TextBox(text_box_ax, label="ACT")

            temp_act = [0]

            def onselect(xmin, xmax):
                # Extract the indices of the selected data range
                indmin, indmax = np.searchsorted(t, (xmin, xmax))

                # Extract selected data
                t_sel = t[indmin:indmax]
                log_acf_sel = log_acf[indmin:indmax]

                # Weigh the fit by the inverse of the variances
                p = np.polyfit(t_sel, log_acf_sel, 1)

                k, b = p
                act = -1 / k
                temp_act[0] = act

                result = f"{act:.3} ps"
                text_box.set_val(result)

                # Plot the fitted curve on ax1
                y_fit = np.polyval(p, t_sel)
                exp_y_fit = np.exp(y_fit)
                fit_line.set_data(t_sel, y_fit)
                exp_fit_line.set_data(t_sel, exp_y_fit)
                act_line.set_data([act, act], [-1, 1])
                fig.canvas.draw_idle()

            span = SpanSelector(ax2, onselect, "horizontal", minspan=2, useblit=True, interactive=True,
                                props={"facecolor": "red", "alpha": 0.2})

            def on_auto_adjust(event):
                margin_ratio = 0.1
                margin_x = t[plot_until - 1] * margin_ratio
                x_min = t[0] - margin_x
                x_max = t[plot_until - 1] + margin_x
                ax2.set_xlim(x_min, x_max)
                margin_y = (log_acf.max() - log_acf.min()) * margin_ratio
                y_min = log_acf.min() - margin_y
                y_max = log_acf.max() + margin_y
                ax2.set_ylim(y_min, y_max)

            button_auto_adjust_ax = fig.add_axes((0.35, 0.01, 0.1, 0.05))
            button_auto_adjust = Button(button_auto_adjust_ax, "Auto Adjust")
            button_auto_adjust.on_clicked(on_auto_adjust)

            def on_reset(event):
                ax1.autoscale()

            button_reset_ax = fig.add_axes((0.5, 0.01, 0.1, 0.05))
            button_reset = Button(button_reset_ax, "Reset")
            button_reset.on_clicked(on_reset)

            def on_next(event):
                acf_dict[job_name] = temp_act[0]
                save_figure(fig, figure_save_dir / f"acf_detail/acf_{job_name}.png")
                exit_flag[0] = False
                plt.close(fig)

            exit_flag = [True]
            button_next_ax = fig.add_axes((0.65, 0.01, 0.1, 0.05))
            button_next = Button(button_next_ax, "Save&Next")
            button_next.on_clicked(on_next)

            def on_exit(event):
                plt.close(fig)

            button_exit_ax = fig.add_axes((0.8, 0.01, 0.1, 0.05))
            button_exit = Button(button_exit_ax, "Exit")
            button_exit.on_clicked(on_exit)

            plt.show()
            if exit_flag[0]:
                break
            else:
                continue
        self.dataset.update_autocorr_time(acf_dict)
        self.dataset.save_act()

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


def main():
    ...


if __name__ == "__main__":
    main()
