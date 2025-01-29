# Author: Mian Qin
# Date Created: 2024/11/13
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils_plot import create_fig_ax, save_figure


class TrackSubclass(type):
    subclass = []

    def __new__(cls, name, bases, namespace):
        new_class = super().__new__(cls, name, bases, namespace)
        new_class.instances = []
        return new_class

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls.instances.append(instance)
        return instance


class PhysicalQuantity(metaclass=TrackSubclass):
    name = ""
    latex_name = ""
    unit = ""

    def __init__(self, model, value):
        self.model = model
        self._value = value

    @property
    def value(self):
        return self._value

    def __str__(self):
        return f"{self.value} ({self.unit})"

    @classmethod
    def plot_all(cls):
        title = "Comparison"
        x_label = "Model"
        y_label = fr"${cls.latex_name}\ (\mathrm{{{cls.unit}}})$"
        fig, ax = create_fig_ax(title, x_label, y_label)

        models = []
        values = []
        for instance in cls.instances:
            models.append(instance.model)
            values.append(instance.value)

        cmap = mpl.colormaps.get_cmap("tab10")
        colors = cmap(np.arange(len(models)))

        ax.bar(models, values, color=colors)
        max_value = max(values)
        min_value = min(values)
        margin = 0.1 * (max_value - min_value)
        ax.set_ylim([min_value - margin, max_value + margin])

        save_path = Path(f"./figure/{cls.name}_compare.png")
        save_figure(fig, save_path)
        plt.close(fig)


class TDependentQuantity(metaclass=TrackSubclass):
    name = ""
    latex_name = ""
    unit = ""

    def __init__(self, model: str, valid_range: tuple[float, float], value_function):
        self.model = model
        self.valid_range = valid_range
        self.value_function = value_function

    def __call__(self, T: float):
        self.check_T_range(T)
        return self.__value_at(T)

    def __value_at(self, T):
        return self.value_function(T)

    def check_T_range(self, T):
        T_min, T_max = self.valid_range
        if T < T_min or T > T_max:
            raise ValueError(f"Temperature {T} is out of the valid range ({T_min}, {T_max})")

    def plot_T_dependent_on_ax(self, ax):
        T_min, T_max = self.valid_range
        T = np.linspace(T_min, T_max, 1000)
        value = [self.__value_at(x) for x in T]
        ax.plot(T, value, "-", label=self.model)

    def plot_T_dependent(self):
        title = self.model
        x_label = r"$T\ (\mathrm{K})$"
        y_label = fr"${self.latex_name}\ (\mathrm{{{self.unit}}})$"
        fig, ax = create_fig_ax(title, x_label, y_label)

        self.plot_T_dependent_on_ax(ax)

        save_path = Path(f"./figure/{self.name}_{self.model}.png")
        save_figure(fig, save_path)

    @classmethod
    def plot_all(cls):
        title = "Comparison"
        x_label = r"$T\ (\mathrm{K})$"
        y_label = fr"${cls.latex_name}\ (\mathrm{{{cls.unit}}})$"
        fig, ax = create_fig_ax(title, x_label, y_label)

        for instance in cls.instances:
            instance.plot_T_dependent_on_ax(ax)

        ax.legend()
        save_path = Path(f"./figure/{cls.name}_compare.png")
        save_figure(fig, save_path)
        plt.close(fig)


def main():
    ...


if __name__ == "__main__":
    main()
