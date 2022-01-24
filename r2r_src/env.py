""" Batched Room-to-Room navigation environment """

import sys
from typing import List, Tuple

metadata_path = "./metadata_parser"
if metadata_path not in sys.path:
    sys.path.append(metadata_path)
from parse_house_segmentations import HouseSegmentationFile
from obj_utils import get_obj_coords

sys.path.append("buildpy36")
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
import h5py
import torch
import torchvision
from param import args
import h5py

from utils import load_datasets, load_nav_graphs, Tokenizer

csv.field_size_limit(sys.maxsize)


class EnvBatch:
    """A simple wrapper for a batch of MatterSim environments,
    using discretized viewpoints and pretrained features"""

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:  # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print("The feature size is %d" % self.feature_size, flush=True)
            elif isinstance(feature_store, h5py.File):
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                scan = list(self.features.keys())[0]
                view = list(self.features[scan].keys())[0]
                self.feature_size = self.features[scan][view].shape[-1]
                print("The feature size is %d" % self.feature_size, flush=True)
        else:
            print("Image features not provided", flush=True)
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60

        if args.reduced_envs:
            self.featurized_scans = args.reduced_env_ids
        else:
            self.featurized_scans = set(self.features.keys())
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)  # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.initialize()
            self.sims.append(sim)

        if args.include_objs:
            self.object_feat_store = h5py.File(args.OBJECT_FEATURES, "r")
            # self.object_info_store = h5py.File(args.OBJECT_PROPOSALS, "r")
            with open(args.OBJECT_CLASS_FILE, "r") as f:
                self.object_classes = json.load(f)
        self.first_iter_done = False

    def _make_id(self, scanId, viewpointId):
        return scanId + "_" + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i, flush=True)
            # sys.stdout.flush()
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getViewpointObjects(
        self, scanId: str, viewpointId: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Get the list of object features in the current viewpoint.
        :param scanId:
        :param viewpointId:
        :return: Tensor with the features of each object in the viewpoint
        """

        feat_store = self.object_feat_store[f"{scanId}/{viewpointId}"]
        # if not self.first_iter_done:
        #     self.first_iter_done = True
        #     print(feat_store.keys(), flush=True)

        roi = feat_store["features"][()]
        orients = feat_store["orientations"][()]
        bboxs = feat_store["bboxs"][()]
        names = feat_store["names"][()]

        return (
            torch.from_numpy(roi),
            torch.from_numpy(orients),
            torch.from_numpy(bboxs),
            [name.decode("utf-8") for name in names],
        )

        # WIDTH = 640
        # HEIGHT = 480
        # pos_tensor = []
        # orient_tensor = []
        # obj_names = []

        # object_ds = self.object_info_store[scanId][viewpointId]

        # for ix in range(36):
        #     for (heading, elevation), bin_name in zip(object_ds["orientations"], object_ds["names"]):
        #         heading, elevation = float(heading), float(elevation)
        #         category = bin_name.decode("utf-8")

        #         x, y = get_obj_coords(ix, elevation, heading, WIDTH=WIDTH, HEIGHT=HEIGHT)

        #         if x is None or y is None:
        #             continue

        #         x_roi, y_roi = x // 32, y // 32

        #         pos_tensor.append([ix, x_roi, y_roi])
        #         obj_names.append(category)

        #         orient_tensor.append([heading, elevation])
        # orient_tensor = torch.from_numpy(np.array(orient_tensor))

        # roi = []
        # valid_indices = []
        # for i, (view, x1, y1) in enumerate(pos_tensor):
        #     x2 = x1 + 1
        #     y2 = y1 + 1
        #     if x2 >= WIDTH // 32 or y2 >= HEIGHT // 32:
        #         continue
        #     valid_indices.append(i)
        #     roi.append(feat[view, :, y1 : y2 + 1, x1 : x2 + 1])

        # if len(roi) == 0:
        #     return torch.zeros((0, 2048, 2, 2)), torch.zeros((0, 2)), torch.zeros((0, 3)), []

        # if not self.first_iter_done:
        #     self.first_iter_done = True
        #     print(
        #         f"First object processinig done. Num objects processed for viewpoint id {viewpointId}: {len(roi)}",
        #         flush=True,
        #     )

        # obj_names = [obj_names[i] for i in valid_indices]
        # valid_indices = torch.LongTensor(valid_indices)
        # roi = torch.stack(roi, dim=0)
        # pos_tensor = torch.Tensor(pos_tensor).index_select(0, valid_indices)
        # orient_tensor = orient_tensor.index_select(0, valid_indices)

        # return roi, orient_tensor, pos_tensor, obj_names

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[state.scanId][state.location.viewpointId]  # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        """Take an action using the full state dependent action interface (with batched input).
        Every action element should be an (index, heading, elevation) tuple."""
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class R2RBatch:
    """Implements the Room to Room navigation task, using discretized viewpoints and pretrained features"""

    def __init__(
        self,
        feature_store,
        batch_size=100,
        seed=10,
        splits=["train"],
        tokenizer=None,
        name=None,
    ):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j, instr in enumerate(item["instructions"]):
                    if item["scan"] not in self.env.featurized_scans:  # For fast training
                        continue
                    new_item = dict(item)
                    new_item["instr_id"] = "%s_%d" % (item["path_id"], j)
                    new_item["instructions"] = instr
                    if tokenizer:
                        new_item["instr_encoding"] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item["instr_encoding"] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item["scan"])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}
        self.buffered_obj_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print("R2RBatch loaded with %d instructions, using splits: %s" % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print("Loading navigation graphs for %d scans" % len(self.scans), flush=True)
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix : self.ix + batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[: self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        """Reset the data index to beginning of epoch. Primarily for testing.
        You must still call reset() for a new episode."""
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        """Determine next action on the shortest path to goal, for supervised training."""
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId  # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        base_heading = (viewId % 12) * math.radians(30)  #! Heading del agente
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)

        if long_id not in self.buffered_state_dict:

            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading  #! Heading del view, relativo al agente
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    if loc.viewpointId not in adj_dict or distance < adj_dict[loc.viewpointId]["distance"]:
                        angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                        adj_dict[loc.viewpointId] = {
                            "heading": loc_heading,  #! Heading de la conexion, relativa al agente
                            "elevation": loc_elevation,  #! Elevacion de la conexion, relativo a la vista mas cercana
                            "normalized_heading": state.heading
                            + loc.rel_heading,  #! Heading de la conexion, relativo a la vista mas cercana
                            "scanId": scanId,
                            "viewpointId": loc.viewpointId,  # Next viewpoint id
                            "pointId": ix,  #! Indice de la vista mas cercana a la conexion
                            "distance": distance,  #! Usado para elegir la vista mas cercana
                            "idx": j + 1,
                            "feature": np.concatenate(
                                (visual_feat, angle_feat), -1
                            ),  #! Features visuales + angulares de conexion
                            # * === Include objects in obs ===
                            "angle_feat": angle_feat,
                            # * ==============================
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {
                    key: c[key]
                    for key in [
                        "normalized_heading",
                        "elevation",
                        "scanId",
                        "viewpointId",
                        "pointId",
                        "idx",
                    ]
                }
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new["pointId"]
                normalized_heading = c_new["normalized_heading"]
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new["heading"] = loc_heading
                angle_feat = utils.angle_feature(c_new["heading"], c_new["elevation"])
                c_new["feature"] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop("normalized_heading")

                # * ======
                c_new["angle_feat"] = angle_feat
                # * ======
                candidate_new.append(c_new)

            return candidate_new

    def make_objs(self, scanId: str, viewpointId: str, agent_head: float, agent_el: float) -> dict:
        if not args.include_objs:
            return {}

        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id in self.buffered_obj_dict:
            o_new = self.buffered_obj_dict[long_id].copy()

            obj_orients_norm = o_new["orients_normalized"]

            original_orients_agent = obj_orients_norm.clone()
            original_orients_agent[:, 0] -= agent_head
            original_orients_agent[:, 1] -= agent_el
            o_new["orients"] = torch.Tensor(
                [utils.angle_feature(head, elev) for (head, elev) in original_orients_agent]
            )
            o_new.pop("orients_normalized")
            return o_new

        roi, orient_tensor, pos_tensor, obj_names = self.env.getViewpointObjects(scanId, viewpointId)

        original_orients_agent = orient_tensor.clone()
        original_orients_agent[:, 0] -= agent_head
        original_orients_agent[:, 1] -= agent_el
        orient_feat = torch.Tensor([utils.angle_feature(head, elev) for (head, elev) in original_orients_agent])

        obj_dict = {
            "feats": roi,
            "bboxs": pos_tensor,
            "names": obj_names,
            "class_ids": torch.LongTensor([self.env.object_classes.index(obj_name) for obj_name in obj_names]),
            "original_orient": original_orients_agent,
        }

        self.buffered_obj_dict[long_id] = {
            **obj_dict,
            "orients_normalized": orient_tensor,
        }

        obj_dict["orients"] = orient_feat

        return obj_dict

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]  #! Saca info de la instruccion
            base_view_id = state.viewIndex  #! Indice de imagen que esta mirando el agente (orientacion)

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # Objects
            objects = self.make_objs(state.scanId, state.location.viewpointId, state.heading, state.elevation)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate(
                (feature, self.angle_feature[base_view_id]), -1
            )  #! Se agrega feats angulares de imagen mirada

            obs.append(
                {
                    "instr_id": item["instr_id"],
                    "scan": state.scanId,
                    "viewpoint": state.location.viewpointId,
                    "viewIndex": state.viewIndex,
                    "heading": state.heading,
                    "elevation": state.elevation,
                    "feature": feature,
                    "candidate": candidate,  #! Conexiones
                    "navigableLocations": state.navigableLocations,
                    "instructions": item["instructions"],
                    "teacher": self._shortest_path_action(state, item["path"][-1]),
                    "path_id": item["path_id"],
                    "objects": objects,  # { names, orients, feats, bboxs }
                }
            )
            if "instr_encoding" in item:
                obs[-1]["instr_encoding"] = item["instr_encoding"]
            # A2C reward. The negative distance between the state and the final state
            obs[-1]["distance"] = self.distances[state.scanId][state.location.viewpointId][item["path"][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        """Load a new minibatch / episodes."""
        if batch is None:  # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:  # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[: len(batch)] = batch
            else:  # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item["scan"] for item in self.batch]
        viewpointIds = [item["path"][0] for item in self.batch]
        headings = [item["heading"] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        """Take action (same interface as makeActions)"""
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum["instructions"]))
            path += self.distances[datum["scan"]][datum["path"][0]][datum["path"][-1]]
        stats["length"] = length / len(self.data)
        stats["path"] = path / len(self.data)
        return stats
