# # Constants


IALAB_MEMBER = True
IALAB_USER = "mrearle"


# # Imports


import sys


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


import sys
import MatterSim
import numpy as np
import networkx as nx


from parse_house_segmentations import HouseSegmentationFile


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


def get_objects(scan, viewpoint):

    base_cache_path = f"/home/{IALAB_USER}/repos/360-visualization/metadata_parser/house_cache"
    HouseSegmentationFile.base_cache_path = base_cache_path

    metadata = HouseSegmentationFile.load_mapping(scan)

    objects = metadata.angle_relative_viewpoint_objects(viewpoint)

    connectivity_path = f"connectivity/{scan}_connectivity.json"

    reachable_viewpoints = metadata.angle_relative_reachable_viewpoints(viewpoint, connectivity_path)
    return objects, reachable_viewpoints


def get_panorama(scan, viewpoint, viewpoint_heading):
    # Get panorama image
    images = []
    for viewpoint_elevation in (np.pi / 2 * x for x in range(-1, 2)):
        im = visualize_panorama_img(scan, viewpoint, viewpoint_heading, viewpoint_elevation)
        images.append(im)

    return np.concatenate(images[::-1])
