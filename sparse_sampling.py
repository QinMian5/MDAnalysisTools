# Author: Mian Qin
# Date Created: 6/4/24
import json

from pathlib import Path
import numpy as np
import scipy.constants as c
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt

from utils import OPDataset


class SparseSampling:
    def __init__(self, dataset: OPDataset, op: str):
        self.dataset = dataset
        self.op = op
        self.dF_phi_dx: None | list[np.ndarray, np.ndarray] = None
        self.F_phi: None | list[np.ndarray, np.ndarray] = None
        self.F_nu_phi: None | list[np.ndarray, np.ndarray] = None
        self.U_phi: None | list[np.ndarray, np.ndarray] = None
        self.F_nu: None | list[np.ndarray, np.ndarray] = None

    def _calculate_dF_phi_dx(self) -> None:
        column_name = self.op
        x_list = []
        df_phi_dx_list = []
        for job_name, data in self.dataset.items():
            params = data.params[column_name]
            op_values = data.df[column_name].values
            assert params["TYPE"] in ["parabola"], f"Unknown bias potential type: {params['type']}"
            if params["TYPE"] == "parabola":
                op_center = params["CENTER"]
                kappa = params["KAPPA"]
                dF_phi_dx = kappa * (op_center - op_values.mean())
                x_list.append(op_center)
                df_phi_dx_list.append(dF_phi_dx)
        self.dF_phi_dx = [np.array(x_list), np.array(df_phi_dx_list)]

    def _integrate(self) -> None:
        x, dF_phi_dx = self.dF_phi_dx
        F_phi = cumulative_simpson(dF_phi_dx, x=x, initial=0)
        self.F_phi = [x, F_phi]

    def f_F_phi(self, x):
        if self.F_phi is None:
            raise RuntimeError("F_phi has not been calculated. Please execute self.calculate() before calling this "
                               "method.")
        cs = CubicSpline(*self.F_phi)
        return cs(x)

    def _calculate_F_nu_phi(self) -> None:
        column_name = self.op
        n_sigma = 0.1
        x_list = []
        F_nu_phi_list = []
        for job_name, data in self.dataset.items():
            op_values = data.df[column_name].values
            mean = op_values.mean()
            std = op_values.std()
            N_total = len(op_values)
            N_center = len(op_values[(op_values <= mean + std * n_sigma) & (op_values >= mean - std * n_sigma)])
            p_center = N_center / N_total / (2 * std * n_sigma)
            F_nu_phi = -1 / data.beta * np.log(p_center)
            # F_nu_phi = 0.5 * np.log(2 * np.pi * std ** 2) / data.beta
            F_nu_phi = F_nu_phi / 1000 * c.N_A  # To kJ/mol
            x_list.append(mean)
            F_nu_phi_list.append(F_nu_phi)
        self.F_nu_phi = [np.array(x_list), np.array(F_nu_phi_list)]

    def _calculate_U_phi(self) -> None:
        column_name = self.op
        x, _ = self.F_nu_phi
        U_phi_list = []
        for i, (job_name, data) in enumerate(self.dataset.items()):
            U_phi = data.calculate_bias_potential({column_name: x[i]})
            U_phi_list.append(U_phi)
        U_phi = np.array(U_phi_list)
        self.U_phi = [x, U_phi]

    def _calculate_F_nu(self) -> None:
        x, F_nu_phi = self.F_nu_phi
        _, U_phi = self.U_phi

        F_phi = self.f_F_phi(x)

        F_nu = F_nu_phi - U_phi + F_phi
        self.F_nu = [x, F_nu]

    def calculate(self):
        self._calculate_dF_phi_dx()
        self._integrate()
        self._calculate_F_nu_phi()
        self._calculate_U_phi()
        self._calculate_F_nu()

    def plot(self, save_fig: bool = True, save_dir=Path("./figure"), k=0.0):
        plt.style.use("presentation.mplstyle")
        column_name = self.op
        fig, ax = plt.subplots()
        x, F_nu = self.F_nu
        line1 = ax.plot(x, F_nu, "bo-", label=r"$F_{\nu} + kx$")
        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel(r"$F$ (kJ/mol)")
        ax.tick_params(axis='y')

        lines = [line1[0]]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels)

        plt.title(f"Sparse Sampling, k = {k}")
        if save_fig:
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"SparseSampling_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()

    def plot_debug(self, save_fig: bool = True, save_dir=Path("./figure")):
        plt.style.use("presentation.mplstyle")
        column_name = self.op
        fig, ax1 = plt.subplots()
        ax1.set_title("Sparse Sampling (Debug)")
        ax1.set_xlabel(f"{column_name}")
        ax1.set_ylabel(r"$F$ (kJ/mol)", color="blue")
        ax1.tick_params(axis='y', colors="red")
        x, F_nu_phi = self.F_nu_phi
        line1 = ax1.plot(x, F_nu_phi, "bo--", label=r"$F_{\nu}^{\phi}$")
        x, U_phi = self.U_phi
        line2 = ax1.plot(x, U_phi, "bv--", label=r"$U_{\phi}$")
        # x = np.linspace(x.min(), x.max(), 100)
        # F_phi = self.f_F_phi(x)
        # line3 = ax1.plot(x, F_phi, "b--", label=r"$F_{\phi}$")

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$d F_{\phi} / d x$", color="red")
        ax2.tick_params(axis='y', colors="red")
        x, dF_phi_dx = self.dF_phi_dx
        line4 = ax2.plot(x, dF_phi_dx, "ro--", label=r"$d F_{\phi} / d x$")

        lines = [line1[0], line2[0], line4[0]]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels)

        if save_fig:
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"SparseSamplingDebug_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()

    def save_result(self, save_dir=Path(".")):
        x, F_nu = self.F_nu
        df = pd.DataFrame({"x": x, "F_nu": F_nu})
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "result_sparse_sampling.csv"
        df.to_csv(save_path)
        print(f"Saved the result to {save_path.resolve()}")

    def load_result(self, save_dir=Path(".")):
        save_path = save_dir / "result_sparse_sampling.csv"
        df = pd.read_csv(save_path)
        x = df["x"].values
        F_nu = df["F_nu"].values
        self.F_nu = [x, F_nu]
        print(f"Loaded the result from {save_path.resolve()}")


def main():
    ...


if __name__ == "__main__":
    main()
