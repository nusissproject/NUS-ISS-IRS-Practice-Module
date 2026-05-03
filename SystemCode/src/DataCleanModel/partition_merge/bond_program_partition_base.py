from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, inf
from typing import List, Sequence, Tuple
from copy import deepcopy

IMG_REAL_MAX = inf


@dataclass
class Coord3D:
    x: float
    y: float
    z: float = 0.0


@dataclass
class BoxInfo:
    rco3d_ul_corner: Coord3D
    rco3d_lr_corner: Coord3D


@dataclass
class PartitionBoxItemInfo:
    box_info: List[BoxInfo]

    @property
    def total_box_num(self) -> int:
        return len(self.box_info)


@dataclass
class RegionInfo:
    rco_ul_corner: Coord3D
    rco_lr_corner: Coord3D
    box_index_list: List[int]

    @property
    def single_box_num(self) -> int:
        return len(self.box_index_list)

    def reset(self) -> None:
        self.rco_ul_corner = Coord3D(0.0, 0.0, 0.0)
        self.rco_lr_corner = Coord3D(IMG_REAL_MAX, IMG_REAL_MAX, 0.0)
        self.box_index_list.clear()


@dataclass
class PartitionXYBoxSingleResult:
    rco_ul_corner: Coord3D
    rco_lr_corner: Coord3D
    box_index_list: List[int]

    @property
    def single_box_num(self) -> int:
        return len(self.box_index_list)

    def clear(self) -> None:
        self.rco_ul_corner = Coord3D(0.0, 0.0, 0.0)
        self.rco_lr_corner = Coord3D(0.0, 0.0, 0.0)
        self.box_index_list.clear()



class PartitionXYBoxResultOverall:
    def __init__(self):
        self.single_boxes: List[PartitionXYBoxSingleResult] = []
        self.box_num: int = 0


class PartitionDebugInfo:
    def __init__(self):
        self.debug_flag: bool = False
        self.log_path: str = ""  # kept for parity with the C++ signature


def itf_get_box_position(rco3d_corner: Coord3D, use_x_coord: bool) -> float:
    return rco3d_corner.x if use_x_coord else rco3d_corner.y


def itf_sort_box(pst_input: PartitionBoxItemInfo, use_x_coord: bool, region: RegionInfo) -> RegionInfo:
    for i in range(region.single_box_num - 1):
        for j in range(region.single_box_num - i - 1):
            box_id1 = region.box_index_list[j]
            box_id2 = region.box_index_list[j + 1]
            pos1 = itf_get_box_position(pst_input.box_info[box_id1].rco3d_ul_corner, use_x_coord)
            pos2 = itf_get_box_position(pst_input.box_info[box_id2].rco3d_ul_corner, use_x_coord)
            # swap two boxes
            if pos1 > pos2:
                region.box_index_list[j], region.box_index_list[j + 1] = (
                    region.box_index_list[j + 1],
                    region.box_index_list[j],
                )
    return region


def itf_compute_box_merge_area(box1: BoxInfo, box2: BoxInfo, fov: Coord3D) -> float:
    rco_ul = Coord3D(
        x=min(box1.rco3d_ul_corner.x, box2.rco3d_ul_corner.x),
        y=min(box1.rco3d_ul_corner.y, box2.rco3d_ul_corner.y),
    )
    rco_lr = Coord3D(
        x=max(box1.rco3d_lr_corner.x, box2.rco3d_lr_corner.x),
        y=max(box1.rco3d_lr_corner.y, box2.rco3d_lr_corner.y),
    )
    width = rco_lr.x - rco_ul.x
    height = rco_lr.y - rco_ul.y
    return width * height if width < fov.x and height < fov.y else IMG_REAL_MAX


def itf_merge_region(region_list: List[RegionInfo], from_idx: int, to_idx: int) -> List[RegionInfo]:
    dst = region_list[to_idx]
    src = region_list[from_idx]
    dst.rco_ul_corner = Coord3D(
        x=min(dst.rco_ul_corner.x, src.rco_ul_corner.x),
        y=min(dst.rco_ul_corner.y, src.rco_ul_corner.y),
    )
    dst.rco_lr_corner = Coord3D(
        x=max(dst.rco_lr_corner.x, src.rco_lr_corner.x),
        y=max(dst.rco_lr_corner.y, src.rco_lr_corner.y),
    )
    dst.box_index_list.extend(src.box_index_list)
    src.reset()
    region_list[to_idx] = dst
    region_list[from_idx] = src
    return region_list


def itf_compute_two_region_distance(
    region_list: Sequence[RegionInfo],
    box_merge_area: Sequence[Sequence[float]],
    fov: Coord3D,
    region1_idx: int,
    region2_idx: int,
) -> float:
    region1 = region_list[region1_idx]
    region2 = region_list[region2_idx]
    rco_ul = Coord3D(
        x=min(region1.rco_ul_corner.x, region2.rco_ul_corner.x),
        y=min(region1.rco_ul_corner.y, region2.rco_ul_corner.y),
    )
    rco_lr = Coord3D(
        x=max(region1.rco_lr_corner.x, region2.rco_lr_corner.x),
        y=max(region1.rco_lr_corner.y, region2.rco_lr_corner.y),
    )
    width = rco_lr.x - rco_ul.x
    height = rco_lr.y - rco_ul.y
    if width >= fov.x and height >= fov.y:
        return IMG_REAL_MAX

    distance = IMG_REAL_MAX
    for box_idx1 in region1.box_index_list:
        for box_idx2 in region2.box_index_list:
            candidate = box_merge_area[box_idx1][box_idx2]
            if candidate < distance:
                distance = candidate
    return distance


def itf_update_region_list(
    region_list: List[RegionInfo],
    region_merge_area: List[List[float]],
    total_box_num: int,
    from_idx: int,
    to_idx: int,
) -> Tuple[List[RegionInfo],List[List[float]]]:
    region_list = itf_merge_region(region_list, from_idx, to_idx)
    for j in range(total_box_num):
        region_merge_area[from_idx][j] = IMG_REAL_MAX
        region_merge_area[to_idx][j] = IMG_REAL_MAX
        region_merge_area[j][from_idx] = IMG_REAL_MAX
        region_merge_area[j][to_idx] = IMG_REAL_MAX
    return region_list, region_merge_area


def itf_update_region_distance(
    region_list: Sequence[RegionInfo],
    region_merge_area: List[List[float]],
    box_merge_area: Sequence[Sequence[float]],
    total_box_num: int,
    fov: Coord3D,
    update_idx: int,
) -> tuple[float, int, int]:
    best_distance = IMG_REAL_MAX
    best_from = 0
    best_to = 0
    for i in range(total_box_num - 1):
        for j in range(i + 1, total_box_num):
            if (
                (i == update_idx or j == update_idx)
                and region_list[i].single_box_num > 0
                and region_list[j].single_box_num > 0
            ):
                region_merge_area[i][j] = itf_compute_two_region_distance(
                    region_list, box_merge_area, fov, i, j
                )
            if region_merge_area[i][j] < best_distance:
                best_distance = region_merge_area[i][j]
                best_to, best_from = i, j
    return best_distance, best_from, best_to


def itf_chc_partition_post_process(
    pst_input: PartitionBoxItemInfo,
    region_list: List[RegionInfo],
    fov: Coord3D,
) -> PartitionXYBoxResultOverall:
    overall = PartitionXYBoxResultOverall()

    for region in region_list:
        if region.single_box_num == 0:
            continue

        width = region.rco_lr_corner.x - region.rco_ul_corner.x
        height = region.rco_lr_corner.y - region.rco_ul_corner.y
        x_tiles = max(1, int(ceil(width / fov.x)))
        y_tiles = max(1, int(ceil(height / fov.y)))

        if x_tiles == 1 and y_tiles == 1:
            single = PartitionXYBoxSingleResult(
                rco_ul_corner=region.rco_ul_corner,
                rco_lr_corner=region.rco_lr_corner,
                box_index_list=list(region.box_index_list),
            )
            overall.single_boxes.append(single)
            continue
        
        stTmp = PartitionXYBoxSingleResult((0.0,0.0,0.0),(0.0,0.0,0.0),[])
        stTmp.box_index_list = []
        use_x_coord = x_tiles >= y_tiles
        region=itf_sort_box(pst_input, use_x_coord, region)
        box_idx = region.box_index_list[0]
        projection_limit = itf_get_box_position(pst_input.box_info[box_idx].rco3d_ul_corner, use_x_coord) + \
              itf_get_box_position(fov, use_x_coord)

        current_ul = pst_input.box_info[box_idx].rco3d_ul_corner
        current_lr = pst_input.box_info[box_idx].rco3d_lr_corner

        for box_idx in region.box_index_list:
            box = pst_input.box_info[box_idx]
            if itf_get_box_position(box.rco3d_lr_corner, use_x_coord) > projection_limit:
                projection_limit = itf_get_box_position(box.rco3d_ul_corner, use_x_coord) + \
                    itf_get_box_position(fov, use_x_coord)
                stTmp.rco_ul_corner = current_ul
                stTmp.rco_lr_corner = current_lr
                overall.single_boxes.append(deepcopy(stTmp))
                current_ul = box.rco3d_ul_corner
                current_lr = box.rco3d_lr_corner
                stTmp.box_index_list = []
            stTmp.box_index_list.append(box_idx)
            current_ul = Coord3D(
                x=min(current_ul.x, box.rco3d_ul_corner.x),
                y=min(current_ul.y, box.rco3d_ul_corner.y),
            )
            current_lr = Coord3D(
                x=max(current_lr.x, box.rco3d_lr_corner.x),
                y=max(current_lr.y, box.rco3d_lr_corner.y),
            )

        if(len(stTmp.box_index_list) > 0):
            stTmp.rco_ul_corner = current_ul
            stTmp.rco_lr_corner = current_lr
            overall.single_boxes.append(deepcopy(stTmp))
    overall.box_num = len(overall.single_boxes)
    return overall


def itf_chc_partition(
    pst_input: PartitionBoxItemInfo,
    fov: Coord3D,
    debug_info: PartitionDebugInfo | None,
) -> PartitionXYBoxResultOverall:
    total = pst_input.total_box_num
    box_merge_area = [[IMG_REAL_MAX for _ in range(total)] for _ in range(total)]
    region_merge_area = [[IMG_REAL_MAX for _ in range(total)] for _ in range(total)]

    region_list = [
        RegionInfo(
            rco_ul_corner=box.rco3d_ul_corner,
            rco_lr_corner=box.rco3d_lr_corner,
            box_index_list=[idx],
        )
        for idx, box in enumerate(pst_input.box_info)
    ]

    for i in range(total):
        for j in range(total):
            box_merge_area[i][j] = itf_compute_box_merge_area(
                pst_input.box_info[i], pst_input.box_info[j], fov
            )

    best_distance = IMG_REAL_MAX
    from_idx = to_idx = 0
    for i in range(total - 1):
        for j in range(i + 1, total):
            region_merge_area[i][j] = itf_compute_two_region_distance(
                region_list, box_merge_area, fov, i, j
            )
            if region_merge_area[i][j] < best_distance:
                best_distance = region_merge_area[i][j]
                to_idx, from_idx = i, j

    while best_distance < IMG_REAL_MAX:
        region_list, region_merge_area = itf_update_region_list(region_list, region_merge_area, total, from_idx, to_idx)
        best_distance, from_idx, to_idx = itf_update_region_distance(
            region_list, region_merge_area, box_merge_area, total, fov, to_idx
        )

    return itf_chc_partition_post_process(pst_input, region_list, fov)