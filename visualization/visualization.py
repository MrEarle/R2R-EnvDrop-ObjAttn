#!/usr/bin/env python
# coding: utf-8

# # Constants


IALAB_MEMBER = True
IALAB_USER = "mrearle"


# # Imports


import os
import sys

from numpy.lib.function_base import copy


if IALAB_MEMBER:
    matterport_build_path = f"/home/{IALAB_USER}/datasets/Matterport3DSimulator/build"
    metadata_script_path = f"/home/{IALAB_USER}/repos/360-visualization/metadata_parser"
else:
    matterport_build_path = f"/Matterport3DSimulator/build"  # Path to simulator
    metadata_script_path = f"/360-visualization/metadata_parser"  # Path to metadata parser of this repository


if matterport_build_path not in sys.path:
    sys.path.append(matterport_build_path)

if metadata_script_path not in sys.path:
    sys.path.append(metadata_script_path)


import json
import sys
import MatterSim
import numpy as np
import networkx as nx
import torch


from parse_house_segmentations import HouseSegmentationFile


# # Simulator


# load navigation graph to calculate the relative heading of the next location
def load_nav_graph(graph_path):
    with open(graph_path) as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        positions[item["image_id"]] = np.array([item["pose"][3], item["pose"][7], item["pose"][11]])
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(item["image_id"], data[j]["image_id"])
        nx.set_node_attributes(G, values=positions, name="position")
    return G


def compute_rel_heading(graph, current_viewpoint, current_heading, next_viewpoint):
    if current_viewpoint == next_viewpoint:
        return 0.0
    target_rel = graph.nodes[next_viewpoint]["position"] - graph.nodes[current_viewpoint]["position"]
    target_heading = np.pi / 2.0 - np.arctan2(target_rel[1], target_rel[0])  # convert to rel to y axis

    rel_heading = target_heading - current_heading
    # normalize angle into turn into [-pi, pi]
    rel_heading = rel_heading - (2 * np.pi) * np.floor((rel_heading + np.pi) / (2 * np.pi))
    return rel_heading


def visualize_panorama_img(scan, viewpoint, heading, elevation):
    WIDTH = 80
    HEIGHT = 480
    pano_img = np.zeros((HEIGHT, WIDTH * 36, 3), np.uint8)
    VFOV = np.radians(55)
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.initialize()
    for n_angle, angle in enumerate(range(-175, 180, 10)):
        sim.newEpisode([scan], [viewpoint], [heading + np.radians(angle)], [elevation])
        state = sim.getState()
        im = state[0].rgb
        im = np.array(im)
        pano_img[:, WIDTH * n_angle : WIDTH * (n_angle + 1), :] = im[..., ::-1]
    return pano_img


def visualize_tunnel_img(scan, viewpoint, heading, elevation):
    WIDTH = 640
    HEIGHT = 480
    VFOV = np.radians(60)
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.initialize()
    sim.newEpisode([scan], [viewpoint], [heading], [elevation])
    state = sim.getState()
    im = np.array(state[0].rgb, copy=True)
    return im[..., ::-1].copy()


# # Metadata Parser


def get_objects(scan, viewpoint):

    base_cache_path = f"/home/{IALAB_USER}/repos/360-visualization/metadata_parser/house_cache"
    HouseSegmentationFile.base_cache_path = base_cache_path

    metadata = HouseSegmentationFile.load_mapping(scan)

    objects = metadata.angle_relative_viewpoint_objects(viewpoint)

    connectivity_path = f"connectivity/{scan}_connectivity.json"

    reachable_viewpoints = metadata.angle_relative_reachable_viewpoints(viewpoint, connectivity_path)
    return objects, reachable_viewpoints


# # Visualization


# Cylinder frame

RADIUS = 5


def data_for_cylinder_along_z(center_x, center_y, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = RADIUS * np.cos(theta_grid) + center_x
    y_grid = RADIUS * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_viewpoint_objs(info, scanId, viewpointId, attn):
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(np.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    def transform_img(im):
        """Prep opencv 3 channel image for the network"""
        im = np.array(im, copy=True)
        im_orig = im.astype(np.float32, copy=True)
        blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
        blob[0, :, :, :] = im_orig[..., ::-1]
        blob = blob / 255.0
        # blob = preprocess(blob)
        return blob

    sample = [int(i) for i in info["obj_sample"]]

    fig, axes = plt.subplots(nrows=3, ncols=12, figsize=(32, 8), gridspec_kw={"wspace": 0.1, "hspace": 0.01})

    bbox_style = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
    start_ix = 0
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [info["heading"] - np.radians(-175)], [np.radians(-30)])
            start_ix = sim.getState()[0].viewIndex
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        # assert state.viewIndex == ix

        # Transform and save generated image
        blob = transform_img(state.rgb)

        _h = (state.viewIndex - start_ix) % 12
        if _h < 0:
            _h += 12

        _e = (state.viewIndex) // 12
        _e = 2 - _e

        ax = axes[_e, _h]
        ax.imshow(blob[0])

        ax.set_xticks([])
        ax.set_yticks([])

        max_avg_attn = np.max(np.average(attn, axis=-1))

        for i, (name, bbox) in enumerate(zip(info["objects"]["names"], info["objects"]["bboxs"])):
            v, x, y, x1, y1 = bbox
            w = x1 - x
            h = y1 - y
            if int(v) != state.viewIndex:
                continue

            x, y, w, h = int(x), int(y), int(w), int(h)

            if i in sample:
                color = cm.viridis(np.average(attn[:, sample.index(i)]) / max_avg_attn)
            else:
                color = "red"
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
            ax.text(x, y - 10, name, color=color, fontsize=6, bbox=bbox_style)

    plt.show()


def plot_matterport_objs_with_traj(info, trajectory, instruction, viewpoint_attn, viewpoint_indices):
    scan = info["scan"]
    viewpoint = info["viewpoint"]
    viewpoint_heading = info["heading"]

    objects, reachable_viewpoints = get_objects(scan, viewpoint)

    images = []
    for viewpoint_elevation in (np.pi / 2 * x for x in range(-1, 2)):
        im = visualize_panorama_img(scan, viewpoint, viewpoint_heading, viewpoint_elevation)
        images.append(im)

    img = np.concatenate(images[::-1])

    plt.figure(figsize=(18, 9))
    plt.imshow(img)
    plt.xticks(np.linspace(0, img.shape[1] - 1, 5), [-180, -90, 0, 90, 180])
    plt.xlabel(f"relative heading from the agent")
    plt.yticks(np.linspace(0, img.shape[0] - 1, 5), [-180, -90, 0, 90, 180])

    bbox_style = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)

    x0, y0 = info["heading"], info["elevation"]
    for obj in objects.itertuples():
        heading, elevation = float(obj.heading), float(obj.elevation)
        category = obj.category_mapping_name

        heading -= x0
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi

        elevation += y0
        while elevation > np.pi:
            heading -= 2 * np.pi
        while elevation < -np.pi:
            elevation += 2 * np.pi

        first_coord = (heading / (2 * np.pi) + 0.5) * img.shape[1]
        second_coord = (0.5 - elevation / (np.pi / 1.1)) * img.shape[0]

        plt.plot(first_coord, second_coord, color="red", marker="v", linewidth=3)
        plt.annotate(category, (first_coord, second_coord), bbox=bbox_style)

    for i, reachable_viewpoint in enumerate(reachable_viewpoints.itertuples()):
        heading, elevation = float(reachable_viewpoint.heading), float(reachable_viewpoint.elevation)

        heading -= x0
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi

        elevation += y0
        while elevation > np.pi:
            heading -= 2 * np.pi
        while elevation < -np.pi:
            elevation += 2 * np.pi

        first_coord = (heading / (2 * np.pi) + 0.5) * img.shape[1]
        second_coord = (0.5 - elevation / (np.pi / 1.1)) * img.shape[0]

        attn_index = viewpoint_indices.index(reachable_viewpoint.name)
        attn_val = float(torch.softmax(viewpoint_attn, dim=-1)[attn_index])
        color = cm.viridis(attn_val)

        plt.plot(
            first_coord,
            second_coord,
            color=color,
            marker="o",
            markersize=50 / reachable_viewpoint.distance,
            linewidth=1,
        )
        plt.annotate(i, (first_coord, second_coord), bbox=bbox_style)
        print(i, reachable_viewpoint.name, reachable_viewpoint.name in trajectory)

    print(instruction)

    plt.show()


def plot_attention(x_labels=None, y_labels=None, attn=None, map_attention=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 20))
    # im = ax.imshow(attn.transpose(1, 0).numpy())

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
    # for i in range(len(x_labels)):
    #     for j in range(len(y_labels)):
    #         text = ax.text(i, j, f"{attn[i, j]:.3f}", ha="center", va="center", color="w")

    ax.set_title("")

    plt.show()


def plot_viewpoint_objs2(info, scanId, viewpointId):
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(np.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    def transform_img(im):
        """Prep opencv 3 channel image for the network"""
        im = np.array(im, copy=True)
        im_orig = im.astype(np.float32, copy=True)
        blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
        blob[0, :, :, :] = im_orig[..., ::-1]
        blob = blob / 255.0
        # blob = preprocess(blob)
        return blob

    fig, axes = plt.subplots(nrows=3, ncols=12, figsize=(32, 8), gridspec_kw={"wspace": 0.1, "hspace": 0.01})

    bbox_style = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
    start_ix = 0
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [info["heading"] - np.radians(-175)], [np.radians(-30)])
            start_ix = sim.getState()[0].viewIndex
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        # assert state.viewIndex == ix

        # Transform and save generated image
        blob = transform_img(state.rgb)

        _h = (state.viewIndex - start_ix) % 12
        if _h < 0:
            _h += 12

        _e = (state.viewIndex) // 12
        _e = 2 - _e

        ax = axes[_e, _h]
        ax.imshow(blob[0])

        ax.set_xticks([])
        ax.set_yticks([])

        view_objects = parse_objs_into_views(info)

        for name, heading, elevation in view_objects[state.viewIndex]:
            h_min = (state.viewIndex % 12) * np.radians(30)
            e_min = (state.viewIndex // 12) * np.radians(30) - np.radians(60)

            x = (heading - h_min) / np.radians(30) * WIDTH
            y = (elevation - e_min) / np.radians(30) * HEIGHT

            print(heading - h_min, elevation - e_min)

            ax.plot(x, y, color="red", marker="v", linewidth=3)
            ax.annotate(name, (x, y), bbox=bbox_style)

    plt.show()


def parse_objs_into_views(info):
    objects, _ = get_objects(info["scan"], info["viewpoint"])

    views = {i: [] for i in range(36)}
    for obj in objects.itertuples():
        heading, elevation = float(obj.heading), float(obj.elevation)
        category = obj.category_mapping_name

        while heading < 0:
            heading += 2 * np.pi
        while heading > 2 * np.pi:
            heading -= 2 * np.pi

        while elevation < -np.pi:
            elevation += 2 * np.pi
        while elevation > np.pi:
            elevation -= 2 * np.pi

        x_view = int((heading / (2 * np.pi)) * 12) % 12
        y_view = int((elevation / (np.pi / 2)) * 3) % 3

        if not (-np.pi / 2 + 2 * y_view * np.pi / 3 <= elevation <= -np.pi / 2 + 2 * (y_view + 1) * np.pi / 3):
            print("weird", (x_view, y_view), (heading, elevation))

        index = x_view + y_view * 12
        views[index].append((category, heading, elevation))

    return views
