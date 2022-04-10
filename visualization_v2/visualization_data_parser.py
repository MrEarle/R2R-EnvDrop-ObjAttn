from collections import defaultdict
from typing import Any, List, Tuple
import torch
import numpy as np
import matplotlib.cm as cm
import pandas as pd


def orientation_to_coord(heading: float, elevation: float, img_height: int, img_width: int) -> Tuple[int, int]:
    first_coord = (heading / (2 * np.pi) + 0.5) * img_width  # img.shape[1]
    if first_coord < 0:
        first_coord += img_width
    second_coord = (0.5 - elevation / (np.pi / 1.1)) * img_height  # img.shape[0]
    return first_coord, second_coord


def parse_objects(
    heading: float,
    elevation: float,
    obj_names: List[str],
    obj_pos: List[Any],
    obj_attn: Any,
    obj_sample: Any,
    img_height: int,
    img_width: int,
    color_map=cm.viridis,
) -> List[dict]:
    x0, y0 = heading, elevation
    obj_attn = torch.sum(obj_attn, dim=0) / obj_attn.shape[0]
    obj_attn = obj_attn / torch.max(obj_attn)

    cat_to_pos_idx = defaultdict(list)
    obj_data = []
    for i, (category, orient) in enumerate(zip(obj_names, obj_pos)):
        (heading, elevation) = float(orient[0]), float(orient[1])

        # Normalize heading and elevation
        # heading -= x0
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi

        elevation += y0
        while elevation > np.pi:
            heading -= 2 * np.pi
        while elevation < -np.pi:
            elevation += 2 * np.pi

        # Determine color if in object sample
        if i in obj_sample:
            color = color_map(obj_attn[obj_sample.index(i)].numpy())
        else:
            color = "black"

        # Determine position
        coord = orientation_to_coord(heading, elevation, img_height, img_width)

        # Determine obj idx
        idx = None
        pos_idx = cat_to_pos_idx[category]
        for pos, ob_id in pos_idx:
            if np.linalg.norm(np.array(pos) - np.array(coord)) < 0.1:
                idx = ob_id
                break

        if idx is None:
            idx = len(pos_idx)
            cat_to_pos_idx[category].append((coord, idx))

        obj_data.append(
            {
                "category": category,
                "coord": coord,
                "color": color,
                "attn": obj_attn[obj_sample.index(i)].numpy() if i in obj_sample else -float("inf"),
                "idx": idx,
            }
        )
    return obj_data


def parse_viewpoints(
    reachable_viewpoints: pd.DataFrame,
    viewpoint_attn: Any,
    viewpoint_indices: Any,
    agent_heading: float,
    agent_elevation: float,
    img_height: int,
    img_width: int,
    color_map=cm.viridis,
) -> List[dict]:
    viewpoint_data = []
    for i, reachable_viewpoint in enumerate(reachable_viewpoints.itertuples()):
        heading, elevation = float(reachable_viewpoint.heading), float(reachable_viewpoint.elevation)

        heading -= agent_heading
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi

        elevation += agent_elevation
        while elevation > np.pi:
            heading -= 2 * np.pi
        while elevation < -np.pi:
            elevation += 2 * np.pi

        attn_index = viewpoint_indices.index(reachable_viewpoint.name)
        attn_val = float(torch.softmax(viewpoint_attn, dim=-1)[attn_index])
        color = color_map(attn_val)

        viewpoint_data.append(
            {
                "name": reachable_viewpoint.name,
                "coord": orientation_to_coord(heading, elevation, img_height, img_width),
                "distance": reachable_viewpoint.distance,
                "color": color,
                "index": i,
            }
        )
    return viewpoint_data
