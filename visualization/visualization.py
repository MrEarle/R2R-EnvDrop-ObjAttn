#!/usr/bin/env python
# coding: utf-8

# # Constants

# In[1]:


IALAB_MEMBER = True
IALAB_USER = "mrearle"


# # Imports

# In[2]:


import os
import sys


# In[3]:


if IALAB_MEMBER:
    matterport_build_path = f"/home/{IALAB_USER}/datasets/Matterport3DSimulator/build"
    metadata_script_path = f"/home/{IALAB_USER}/repos/360-visualization/metadata_parser"
else:
    matterport_build_path = f"/Matterport3DSimulator/build"  # Path to simulator
    metadata_script_path = f"/360-visualization/metadata_parser"  # Path to metadata parser of this repository


# In[4]:


if matterport_build_path not in sys.path:
    sys.path.append(matterport_build_path)

if metadata_script_path not in sys.path:
    sys.path.append(metadata_script_path)


# In[5]:


import json
import sys
import MatterSim
import numpy as np
import networkx as nx


from parse_house_segmentations import HouseSegmentationFile


# # Simulator

# In[6]:


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


# In[7]:


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
    sim.init()
    sim.newEpisode(scan, viewpoint, heading, elevation)
    state = sim.getState()
    im = state.rgb
    return im[..., ::-1].copy()


# # Metadata Parser

# In[8]:


def get_objects(scan, viewpoint):
    # In[9]:

    base_cache_path = f"/home/{IALAB_USER}/repos/360-visualization/metadata_parser/house_cache"
    HouseSegmentationFile.base_cache_path = base_cache_path

    # In[10]:

    metadata = HouseSegmentationFile.load_mapping(scan)

    # In[11]:

    objects = metadata.angle_relative_viewpoint_objects(viewpoint)

    # In[12]:

    connectivity_path = f"connectivity/{scan}_connectivity.json"

    reachable_viewpoints = metadata.angle_relative_reachable_viewpoints(viewpoint, connectivity_path)
    return objects, reachable_viewpoints


# # Visualization

# In[13]:


# Cylinder frame

RADIUS = 5


def data_for_cylinder_along_z(center_x, center_y, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = RADIUS * np.cos(theta_grid) + center_x
    y_grid = RADIUS * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid
