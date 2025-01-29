# Author: Mian Qin
# Date Created: 2/4/24
from pathlib import Path
from collections import OrderedDict

import numpy as np
import scipy.constants as c

from op_dataset import OPDataset


def read_solid_like_atoms(file_path: Path) -> dict[str, list[str]]:
    solid_like_atoms_dict = OrderedDict()
    with open(file_path) as file:
        for line in file:
            line = line.strip().split()
            if len(line) == 0:  # End of file
                break
            t = float(line[0])
            indices = [str(x) for x in line[1:]]
            solid_like_atoms_dict[f"{t:.1f}"] = indices
    return solid_like_atoms_dict


def calculate_histogram_parameters(dataset: OPDataset, column_name, num_bins: int = None,
                                   bin_width: float = None, bin_range: tuple[float, float] = None):
    """
        Calculate parameters for histogram plotting: num_bins and bin_range.

        :param dataset: An instance of the UmbrellaSamplingDataset that stores the simulation data.
        :param column_name: Name of the column to compute the histogram for.
        :param num_bins: Number of bins. If not specified, calculates based on bin_width or default strategy.
        :param bin_width: The bin width for the histogram. Used if num_bins is not specified.
        :param bin_range: A tuple specifying the minimum and maximum range of the bins.
                          If not specified, uses the minimum and maximum range in the data.
        :return: A tuple containing calculated num_bins and bin_range.
    """
    if bin_range is None:
        column_min = min(data.df_prd[column_name].min() for data in dataset.values())
        column_max = max(data.df_prd[column_name].max() for data in dataset.values())
        bin_range = (column_min, column_max)

    if num_bins is None and bin_width is None:
        total_points = sum(len(data.df_prd[column_name]) for data in dataset.values())
        # The default strategy for choosing num_bins ensures an average of 300 points per bin.
        num_bins = max(total_points // 300, 1)
    elif bin_width is not None:
        # Calculate num_bins using bin_width if num_bins is not explicitly provided
        num_bins = int((bin_range[1] - bin_range[0]) / bin_width)

    return num_bins, bin_range


def convert_unit(src_value: float | np.ndarray, src_unit="kJ/mol", dst_unit="kT", T=None) -> float | np.ndarray:
    if T is None:
        T = 300
    _valid_units = ["kJ/mol", "kT"]
    if src_unit not in _valid_units or dst_unit not in _valid_units:
        raise ValueError(f"src_unit and dst_unit must be in {_valid_units}")

    if src_unit == "kJ/mol":
        value_in_SI = src_value * 1000 / c.N_A
    elif src_unit == "kT":
        value_in_SI = src_value * c.k * T
    else:
        raise ValueError(f"Unsupported source unit: {src_unit}")

    if dst_unit == "kJ/mol":
        dst_value = value_in_SI * c.N_A / 1000
    elif dst_unit == "kT":
        dst_value = value_in_SI / (c.k * T)
    else:
        raise ValueError(f"Unsupported destination unit: {src_unit}")
    return dst_value


def calculate_triangle_area(nodes, faces):
    if len(faces) != 0:
        triangles = nodes[faces]

        vec1 = triangles[:, 1, :] - triangles[:, 0, :]
        vec2 = triangles[:, 2, :] - triangles[:, 0, :]
        cross_product = np.cross(vec1, vec2)
        areas = 0.5 * np.linalg.norm(cross_product, axis=1)

        total_area = np.sum(areas)
    else:
        areas = []
        total_area = 0
    return areas, total_area


def main():
    nodes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    faces = np.array([[0, 1, 2], [1, 0, 2], [1, 2, 3]])
    areas, total_area = calculate_triangle_area(nodes, faces)
    print(areas)


if __name__ == "__main__":
    DATA_DIR = Path("/Users/qinmian/Data_unsync/testdata/ModelPotential1d/")
    main()
