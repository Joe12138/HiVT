import os
import json

import numpy as np
import pandas as pd
import torch
import copy

from tqdm import tqdm
from itertools import permutations
from itertools import product
from torch_geometric.data import Data, Dataset
from typing import Callable, Dict, List, Optional, Tuple, Union

from hdmap.hd_map import HDMap
from hdmap.util.map_util import get_lane_id_in_xy_bbox
from dataset.pandas_dataset import DatasetPandas, DATA_DICT

from utils import TemporalData


class InteractionDataset(Dataset):
    def __init__(self,
                 root: str,
                 save_dir: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius

        if split == "sample":
            self._directory = "forecasting_sample"
        elif split == "train":
            self._directory = "train"
        elif split == "val":
            self._directory = "val"
        elif split == "test":
            self._directory = "test"
        else:
            raise ValueError(split + " is not valid")

        self.root = root
        self.save_dir = save_dir
        self._raw_file_names = os.listdir(self.raw_dir)

        self.target_veh_path = f"{self.root}/{self._split}_target_filter"
        self.map_path = f"{self.root}/maps"
        self.data_path = f"{self.root}/{self._split}"

        self.target_veh_list, self.scene_set = self.get_target_veh_list()
        self.map_dict = self.get_map_dict()
        self.dataset_dict = self.get_dataset_dict()

        self._processed_file_names = [f"{scene_name}_{case_id}_{track_id}.pt" for scene_name, case_id, track_id in
                                      self.target_veh_list]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]

        super(InteractionDataset, self).__init__(root, transform=transform)

    def get_target_veh_list(self):
        target_veh_list = []
        map_set = set()
        file_list = os.listdir(self.target_veh_path)
        for file_name in file_list:
            scene_name = file_name[:-5]
            map_set.add(scene_name)
            with open(os.path.join(self.target_veh_path, file_name), "r", encoding="UTF-8") as f:
                target_dict = json.load(f)

                for k in target_dict.keys():
                    case_id = int(k)

                    for track_id in target_dict[k]:
                        target_veh_list.append((scene_name, case_id, track_id))
                f.close()
            break

        return target_veh_list, map_set

    def get_map_dict(self):
        map_dict = {}

        for scene in self.scene_set:
            hd_map = HDMap(osm_file_path=os.path.join(self.map_path, f"{scene}.osm"))
            map_dict[scene] = hd_map

        return map_dict

    def get_dataset_dict(self):
        dataset_dict = {}

        for scene in self.scene_set:
            dataset_pandas = DatasetPandas(data_path=os.path.join(self.data_path, f"{scene}_{self._split}.csv"))
            dataset_dict[scene] = dataset_pandas

        return dataset_dict

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.save_dir, self._directory, "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        for scene_name, case_id, track_id in tqdm(self.target_veh_list):
            case_data = copy.deepcopy(self.dataset_dict[scene_name].get_case_data(case_id=case_id))
            kwargs = process_interaction(split=self._split,
                                         scene_name=scene_name,
                                         case_id=case_id,
                                         track_id=track_id,
                                         df=case_data,
                                         hd_map=self.map_dict[scene_name],
                                         radius=self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs["seq_id"])+".pt"))

    def len(self) -> int:
        return len(self.target_veh_list)

    def get(self, idx: int) -> Data:
        return torch.load(self.processed_paths[idx])


def find_av(case_data: pd.DataFrame, target_id: int) -> int:
    actor_ids = list(case_data["track_id"].unique())

    id_dist_dict = {}
    target_df = case_data[case_data["track_id"] == target_id].values
    target_xy = target_df[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
    for actor_id in actor_ids:
        if actor_id == target_id:
            continue

        actor_df = case_data[case_data["track_id"] == actor_id].values

        if actor_df.shape[0] == 40:
            xy_array = actor_df[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
            diff_array = xy_array-target_xy
            dist = np.sum(np.hypot(diff_array[:, 0], diff_array[:, 1]))

            id_dist_dict[actor_id] = dist

    if len(id_dist_dict) == 0:
        return target_id
    else:
        id_dist_list = sorted(id_dist_dict.items(), key=lambda x: x[1])
        return id_dist_list[0][0]


def get_lane_features(hd_map: HDMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(get_lane_id_in_xy_bbox(query_x=node_position[0],
                                               query_y=node_position[1],
                                               hd_map=hd_map,
                                               query_search_range_manhattan=radius))
    node_positions = torch.matmul(node_positions-origin, rotate_mat).float()

    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(hd_map.id_lane_dict[lane_id].centerline_array).float()
        lane_centerline = torch.matmul(lane_centerline-origin, rotate_mat)
        traffic_control, turn_direction, is_intersection, speed_limit = hd_map.get_lane_info(lane_id)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:]-lane_centerline[:-1])
        count = len(lane_centerline)-1
        is_intersections.append(is_intersection*torch.ones(count, dtype=torch.uint8))

        if turn_direction == 'NONE' or turn_direction is None:
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))

    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors


def process_interaction(split: str,
                        scene_name: str,
                        case_id: int,
                        track_id: int,
                        df: pd.DataFrame,
                        hd_map: HDMap,
                        radius: float) -> Dict:
    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df["frame_id"].unique()))
    historical_timestamps = timestamps[:10]
    historical_df = df[df["frame_id"].isin(historical_timestamps)]
    actor_ids = list(historical_df["track_id"].unique())

    df = df[df["track_id"].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_id = find_av(case_data=df, target_id=track_id)
    av_df = df[df["track_id"] == av_id].iloc
    av_index = actor_ids.index(av_df[0]["track_id"])
    agent_df = df[df["track_id"] == track_id].iloc
    agent_index = actor_ids.index(agent_df[0]["track_id"])

    city = scene_name

    # make the scene centered at AV
    origin = torch.tensor([av_df[9]["x"], av_df[9]["y"]], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[8]["x"], av_df[8]["y"]], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 40, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 40, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 10, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby("track_id"):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['frame_id']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 9]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 10:] = True

        xy = torch.from_numpy(np.stack([actor_df['x'].values, actor_df['y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 10, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 10:] = True

    # bos_mask is True if time step is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 10] = padding_mask[:, :9] & ~padding_mask[:, 1:10]

    positions = x.clone()
    x[:, 10:] = torch.where((padding_mask[:, 9].unsqueeze(-1) | padding_mask[:, 10:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 10:] - x[:, 9].unsqueeze(-2))
    x[:, 1: 10] = torch.where((padding_mask[:, : 9] | padding_mask[:, 1: 10]).unsqueeze(-1),
                              torch.zeros(num_nodes, 9, 2),
                              x[:, 1: 10] - x[:, : 9])
    x[:, 0] = torch.zeros(num_nodes, 2)

    # get lane features at the current time step
    df_9 = df[df["frame_id"] == timestamps[9]]
    node_inds_9 = [actor_ids.index(actor_id) for actor_id in df_9["track_id"]]
    node_position_9 = torch.from_numpy(np.stack([df_9["x"].values, df_9["y"].values], axis=-1)).float()

    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(hd_map=hd_map,
                                             node_inds=node_inds_9,
                                             node_positions=node_position_9,
                                             origin=origin,
                                             rotate_mat=rotate_mat,
                                             radius=radius)

    y = None if split == 'test' else x[:, 10:]
    seq_id = f"{scene_name}_{case_id}_{track_id}"

    return {
        'x': x[:, : 10],  # [N, 10, 2]
        'positions': positions,  # [N, 40, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': seq_id,
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
        "scene_name": scene_name,
        "track_id": track_id,
        "case_id": case_id
    }



