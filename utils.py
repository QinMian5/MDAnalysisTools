# Author: Mian Qin
# Date Created: 2/4/24
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import scipy.constants as c

from op_dataset import OPDataset


def format_uncertainty(value, uncertainty, latex=True):
    std_digits = int(np.floor(np.log10(uncertainty)))
    std_rounded = round(uncertainty, -std_digits + 1)
    mean_rounded = round(value, -std_digits + 1)
    plus_minus_sign = r"\pm" if latex else "±"
    result = f"{mean_rounded:.{-std_digits + 1}f} {plus_minus_sign} {std_rounded:.{-std_digits + 1}f}"
    return result


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


def compute_mean_curvature(nodes, faces):
    normals = np.zeros_like(nodes)
    for face in faces:
        v0, v1, v2 = nodes[face]
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        normals[face[0]] += n
        normals[face[1]] += n
        normals[face[2]] += n
    normals_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_len[normals_len == 0] = 1.0
    normals /= normals_len

    # 构建边到三角形的映射
    edge_to_triangles = defaultdict(list)
    for fidx, face in enumerate(faces):
        for i in range(3):
            a, b = sorted([face[i], face[(i + 1) % 3]])
            edge_to_triangles[(a, b)].append(fidx)

    # 计算 Voronoi 面积
    voronoi_area = np.zeros(len(nodes))
    for face in faces:
        i, j, k = face
        vi, vj, vk = nodes[i], nodes[j], nodes[k]

        # 计算向量和长度
        e_ij = vj - vi
        e_ik = vk - vi
        e_ji = vi - vj
        e_jk = vk - vj
        e_ki = vi - vk
        e_kj = vj - vk

        # 计算各个角的cot值
        def compute_cot(a, b):
            dot = np.dot(a, b)
            cross = np.linalg.norm(np.cross(a, b))
            return dot / (cross + 1e-6) if cross != 0 else 0.0

        cot_i = compute_cot(e_ij, e_ik)
        cot_j = compute_cot(e_ji, e_jk)
        cot_k = compute_cot(e_ki, e_kj)

        # 三角形面积
        area = 0.5 * np.linalg.norm(np.cross(e_ij, e_ik))

        # 判断钝角
        theta_i_obtuse = np.dot(e_ij, e_ik) < 0
        theta_j_obtuse = np.dot(e_ji, e_jk) < 0
        theta_k_obtuse = np.dot(e_ki, e_kj) < 0

        # 计算贡献
        def compute_contrib(vertex, e1, e2, cot_other1, cot_other2, is_obtuse):
            if is_obtuse:
                return 0.5 * area
            else:
                return (np.dot(e1, e1) * cot_other2 + np.dot(e2, e2) * cot_other1) / 8

        contrib_i = compute_contrib(i, e_ij, e_ik, cot_j, cot_k, theta_i_obtuse)
        contrib_j = compute_contrib(j, e_ji, e_jk, cot_k, cot_i, theta_j_obtuse)
        contrib_k = compute_contrib(k, e_ki, e_kj, cot_i, cot_j, theta_k_obtuse)

        voronoi_area[i] += contrib_i
        voronoi_area[j] += contrib_j
        voronoi_area[k] += contrib_k

    # 计算拉普拉斯-贝尔特拉米算子 delta_s
    delta_s = np.zeros_like(nodes)
    neighbors = defaultdict(set)
    for face in faces:
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            neighbors[a].add(b)
            neighbors[b].add(a)

    for vi in range(len(nodes)):
        for vj in neighbors[vi]:
            a, b = sorted([vi, vj])
            edge = (a, b)
            tris = edge_to_triangles.get(edge, [])
            cot_sum = 0.0
            for fidx in tris:
                tri = faces[fidx]
                vk = [v for v in tri if v != vi and v != vj][0]
                e_vi_vk = nodes[vk] - nodes[vi]
                e_vj_vk = nodes[vk] - nodes[vj]
                cot = compute_cot(e_vi_vk, e_vj_vk)
                cot_sum += cot
            delta_s[vi] += cot_sum * (nodes[vj] - nodes[vi])

    # 计算平均曲率
    H = np.zeros(len(nodes))
    for vi in range(len(nodes)):
        A = voronoi_area[vi]
        if A == 0:
            H[vi] = 0.0
            continue
        Hn = delta_s[vi] / (2 * A)
        H[vi] = np.dot(Hn, normals[vi])

    return H


def main():
    nodes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 0, 2], [1, 2, 3]])
    # areas, total_area = calculate_triangle_area(nodes, faces)
    # print(areas)
    H = compute_mean_curvature(nodes, faces)
    print(H)


if __name__ == "__main__":
    DATA_DIR = Path("/Users/qinmian/Data_unsync/testdata/ModelPotential1d/")
    main()
