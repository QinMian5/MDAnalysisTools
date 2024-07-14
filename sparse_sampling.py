# Author: Mian Qin
# Date Created: 6/4/24
import json

from pathlib import Path
import numpy as np
import scipy.constants as c
from scipy.integrate import cumulative_simpson, simps
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt

from utils import OPDataset, convert_unit


class SparseSampling:
    def __init__(self, dataset: None | OPDataset, op: str):
        self.dataset = dataset
        self.op = op
        self.x: None | np.ndarray = None
        self.dF_lambda_dx: None | list[np.ndarray, np.ndarray] = None
        self.F_lambda: None | np.ndarray = None
        self.F_nu_lambda: None | np.ndarray = None
        self.U_lambda: None | np.ndarray = None
        self.F_nu: None | np.ndarray = None
        self.energy: None | list[np.ndarray, np.ndarray] = None

    def _calculate_x(self) -> None:
        column_name = self.op
        x_list = []
        for job_name, data in self.dataset.items():
            op_values = data.df[column_name].values
            mean = op_values.mean()
            x_list.append(mean)
        self.x = np.array(x_list)

    def _calculate_dF_lambda_dx(self) -> None:
        column_name = self.op
        x_star_list = []
        df_lambda_dx_list = []
        for job_name, data in self.dataset.items():
            params = data.params[column_name]
            op_values = data.df[column_name].values
            assert params["TYPE"] in ["parabola"], f"Unknown bias potential type: {params['type']}"
            if params["TYPE"] == "parabola":
                op_center = params["CENTER"]
                kappa = params["KAPPA"]
                dF_phi_dx = kappa * (op_center - op_values.mean())
                x_star_list.append(op_center)
                df_lambda_dx_list.append(dF_phi_dx)
        self.dF_lambda_dx = [np.array(x_star_list), np.array(df_lambda_dx_list)]

    def _calculate_F_lambda(self) -> None:
        x, dF_lambda_dx = self.dF_lambda_dx
        F_lambda = cumulative_simpson(dF_lambda_dx, x=x, initial=0)
        self.F_lambda = F_lambda

    def _calculate_F_nu_lambda(self) -> None:
        column_name = self.op
        F_nu_lambda_list = []
        for job_name, data in self.dataset.items():
            op_values = data.df[column_name].values
            std = op_values.std()
            F_nu_lambda = 0.5 * np.log(2 * np.pi * std ** 2) / data.beta
            F_nu_lambda = F_nu_lambda / 1000 * c.N_A  # To kJ/mol
            F_nu_lambda_list.append(F_nu_lambda)
        self.F_nu_lambda = np.array(F_nu_lambda_list)

    def _calculate_U_lambda(self) -> None:
        column_name = self.op
        x = self.x
        U_lambda_list = []
        for i, (job_name, data) in enumerate(self.dataset.items()):
            U_lambda = data.calculate_bias_potential({column_name: x[i]})
            U_lambda_list.append(U_lambda)
        self.U_lambda = np.array(U_lambda_list)

    def _calculate_F_nu(self) -> None:
        F_nu_lambda = self.F_nu_lambda
        U_lambda = self.U_lambda
        F_lambda = self.F_lambda

        F_nu = F_nu_lambda - U_lambda + F_lambda
        self.F_nu = F_nu

    def _calculate_energy(self) -> None:
        x = self.x
        F_nu = self.F_nu
        energy = F_nu - F_nu.min()
        self.energy = [x, energy]

    def calculate(self):
        self._calculate_x()
        self._calculate_dF_lambda_dx()
        self._calculate_F_lambda()
        self._calculate_F_nu_lambda()
        self._calculate_U_lambda()
        self._calculate_F_nu()
        self._calculate_energy()

    def plot(self, save_fig: bool = True, save_dir=Path("./figure"), delta_mu=0.0):
        plt.style.use("presentation.mplstyle")
        column_name = self.op
        fig, ax = plt.subplots()
        x = self.x
        F_nu = self.F_nu
        line1 = ax.plot(x, convert_unit(F_nu) + delta_mu * x, "bo-")
        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel(r"$\beta F + \Delta\mu N$")
        ax.tick_params(axis='y')

        lines = [line1[0]]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels)

        plt.title(rf"Sparse Sampling, $\Delta\mu = {delta_mu} k_BT$")
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
        ax1.set_ylabel(r"$\beta F$", color="blue")
        ax1.tick_params(axis='y', colors="blue")
        x = self.x
        F_nu_lambda = self.F_nu_lambda
        line1 = ax1.plot(x, convert_unit(F_nu_lambda), "bo--", label=r"$\beta F_{\nu}^{\lambda}$")
        U_lambda = self.U_lambda
        line2 = ax1.plot(x, convert_unit(U_lambda), "bv--", label=r"$\beta U_{\lambda}$")

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\beta dF_{\lambda} / dx$", color="red")
        ax2.tick_params(axis='y', colors="red")
        x, dF_lambda_dx = self.dF_lambda_dx
        line4 = ax2.plot(x, convert_unit(dF_lambda_dx), "ro--", label=r"$\beta dF_{\lambda} / dx$")

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
        x, energy = self.energy
        x_star, dF_lambda_dx = self.dF_lambda_dx
        df = pd.DataFrame({"x": x, "energy": energy, "x_star": x_star, "dF_lambda_dx": dF_lambda_dx})
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "result_sparse_sampling.csv"
        df.to_csv(save_path)
        print(f"Saved the result to {save_path.resolve()}")

    def load_result(self, save_dir=Path(".")):
        save_path = save_dir / "result_sparse_sampling.csv"
        df = pd.read_csv(save_path)
        x = df["x"].values
        energy = df["energy"].values
        x_star = df["x_star"].values
        dF_lambda_dx = df["dF_lambda_dx"].values
        self.energy = [x, energy]
        self.dF_lambda_dx = [x_star, dF_lambda_dx]
        print(f"Loaded the result from {save_path.resolve()}")


def main():
    ...


if __name__ == "__main__":
    main()
