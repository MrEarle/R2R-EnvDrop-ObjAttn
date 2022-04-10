from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .visualization_data_parser import parse_objects, parse_viewpoints
from .visualization_utils import get_objects, get_panorama, visualize_panorama_img

BBOX_STYLE = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.1)


def make_plots(info, viewpoint_attn, candidates, object_attention, instruction):
    plt.close("all")

    # 1. Plot agent current view with instruction and number-to-name mapping
    # Include object attention on shown objects.
    idx_to_name, object_data = plot_view(
        scan=info["scan"],
        viewpoint=info["viewpoint"],
        viewpoint_heading=info["heading"],
        viewpoint_attn=viewpoint_attn,
        viewpoint_indices=candidates,
        agent_heading=info["heading"],
        agent_elevation=info["elevation"],
        object_names=info["objects"]["names"],
        object_pos=info["objects"]["original_orient"],
        object_attn=object_attention,
        object_sample=info["obj_sample"].tolist(),
        instruction=instruction,
    )

    # 2. Plot object attention map and viewpoint attention map
    object_names_with_idx = [f"{object['category']} {object['idx']}" for object in object_data]
    num_objs = sum(info["mask"])
    object_names = [object_names_with_idx[i] for i in info["obj_sample"]][:num_objs]
    viewpoint_names = [idx_to_name[i] for i in candidates]
    plot_attentions(viewpoint_names, viewpoint_attn, object_names, object_attention)


def plot_attentions(viewpoint_indices, viewpoint_attn, obj_names, obj_attn):
    def map_attn(attn, ax):
        ax.imshow(attn.transpose(0, 1).numpy())
        for i in range(len(viewpoint_indices)):
            for j in range(len(obj_names)):
                ax.text(i, j, f"{attn[i, j]:.3f}", ha="center", va="center", color="black")

    def map_viewpoint_attn(attn, ax):
        ax.imshow(viewpoint_attn.unsqueeze(0).numpy())
        for i in range(len(viewpoint_indices)):
            ax.text(
                i,
                0,
                f"{torch.softmax(viewpoint_attn, -1)[i]:.3f}",
                ha="center",
                va="center",
                color="black",
            )

    plot_attention(x_labels=viewpoint_indices, y_labels=obj_names, attn=obj_attn, map_attention=map_attn)
    plot_attention(x_labels=viewpoint_indices, y_labels=None, attn=viewpoint_attn, map_attention=map_viewpoint_attn)


def plot_attention(x_labels=None, y_labels=None, attn=None, map_attention=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
    else:
        ax.set_xticks([])

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticks([])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    map_attention(attn, ax)

    ax.set_title("")
    ax.set_aspect(0.5)

    plt.show()


def plot_view(
    scan: str,
    viewpoint: str,
    viewpoint_heading: float,
    viewpoint_attn: Any,
    viewpoint_indices: Any,
    agent_heading: float,
    agent_elevation: float,
    object_names: List[str],
    object_pos: List[Any],
    object_attn: Any,
    object_sample: Any,
    instruction: str,
):
    _, reachable_viewpoints = get_objects(scan, viewpoint)

    panorama = get_panorama(scan, viewpoint, viewpoint_heading)
    img_height, img_width = panorama.shape[:2]

    object_data = parse_objects(
        agent_heading, agent_elevation, object_names, object_pos, object_attn, object_sample, img_height, img_width
    )
    viewpoint_data = parse_viewpoints(
        reachable_viewpoints, viewpoint_attn, viewpoint_indices, agent_heading, agent_elevation, img_height, img_width
    )

    fig = plt.figure(figsize=(18, 10))
    view_ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))

    # Setup panorama image
    view_ax.imshow(panorama)
    view_ax.set_xticks(np.linspace(0, img_width - 1, 5), [-180, -90, 0, 90, 180])
    view_ax.set_xlabel(f"relative heading from the agent")
    view_ax.set_yticks(np.linspace(0, img_height - 1, 5), [-180, -90, 0, 90, 180])

    # Show objects
    objects = {}  # { label: (x, y, color, attn) }
    for object in object_data:
        x, y = object["coord"]
        obj_idx = object["idx"]
        label = f"{object['category']} {obj_idx}"

        if label not in objects:
            objects[label] = (x, y, object["color"], object["attn"])
        elif object["attn"] > objects[label][3]:
            objects[label] = (x, y, object["color"], object["attn"])

    for label, (x, y, color, _) in objects.items():
        view_ax.annotate(label, (x + 15, y + 15), bbox=BBOX_STYLE, color="black")
        view_ax.plot(x, y, marker="v", color=color, linewidth=3)

    # for object in object_data:
    #     x, y = object["coord"]
    #     obj_idx = object["idx"]
    #     label = f"{object['category']} {obj_idx}"
    #     # x = (x + int(img_width * 0.75)) % img_width
    #     view_ax.annotate(label, (x + 15, y + 15), bbox=BBOX_STYLE, color="black")
    #     view_ax.plot(x, y, marker="v", color=object["color"], linewidth=3)

    # Show viewpoints
    idx_to_name = {}
    for view in viewpoint_data:
        x, y = view["coord"]
        view_ax.annotate(view["index"], (x, y), bbox=BBOX_STYLE, color="black")
        view_ax.plot(x, y, marker="o", color=view["color"], linewidth=1, markersize=50 / view["distance"])
        idx_to_name[view["index"]] = view["name"]

    # Add text
    text = f"Instruction:\n{insert_newlines(instruction)}\n\nIndex to Viewpoint:\n"
    for i, view in idx_to_name.items():
        text += f"{i} -> {view}\n"
    fig.text(0.1, 0.1, text, fontsize=16, wrap=True, verticalalignment="top")

    plt.show()
    idx_to_name["STOP"] = "STOP"
    return {name: idx for idx, name in idx_to_name.items()}, object_data


def insert_newlines(text, every=100):
    words = text.split(" ")
    lines = []
    to_insert = []
    curr_len = 0
    for w in words:
        to_insert.append(w)
        curr_len += len(w)
        if curr_len > every:
            lines.append(" ".join(to_insert))
            to_insert = []
            curr_len = 0

    if to_insert:
        lines.append(" ".join(to_insert))
    return "\n".join(lines)
