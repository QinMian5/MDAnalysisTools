# Author: Mian Qin
# Date Created: 9/18/24
from pathlib import Path

import matplotlib.figure
import matplotlib.axes
import matplotlib.lines
import matplotlib.container
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from PIL.ImageOps import contain


def create_fig_ax(title, x_label, y_label, **kwargs) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig, ax


def plot_with_error_band(ax: matplotlib.axes.Axes, x, y_u, fmt="", n_sigma=2, **kwargs) -> matplotlib.lines.Line2D:
    y = unp.nominal_values(y_u)
    y_err = n_sigma * unp.std_devs(y_u)
    line = ax.plot(x, y, fmt, **kwargs)[0]
    color = line.get_color()
    ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.2)
    return line


def plot_with_error_bar(ax: matplotlib.axes.Axes, x, y_u, fmt="", n_sigma=2, **kwargs) -> matplotlib.container.ErrorbarContainer:
    y = unp.nominal_values(y_u)
    y_err = n_sigma * unp.std_devs(y_u)
    container = ax.errorbar(x, y, yerr=y_err, fmt=fmt, capsize=5,
                capthick=1, elinewidth=1, **kwargs)
    return container


def save_figure(fig: matplotlib.figure.Figure, path: Path):
    save_dir = path.parent
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {path.resolve()}")


def main():
    ...


if __name__ == "__main__":
    main()
