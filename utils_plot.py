# Author: Mian Qin
# Date Created: 9/18/24
from pathlib import Path

import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp


def create_fig_ax(title, x_label, y_label) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig, ax


def plot_with_error_band(ax: matplotlib.axes.Axes, x, y_u, color="blue", label=None):
    y = unp.nominal_values(y_u)
    y_err = unp.std_devs(y_u)
    line = ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.2)
    return line


def save_figure(fig: matplotlib.figure.Figure, path: Path):
    save_dir = path.parent
    save_dir.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {path.resolve()}")


def main():
    ...


if __name__ == "__main__":
    main()
