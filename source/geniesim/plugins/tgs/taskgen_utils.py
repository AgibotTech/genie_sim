# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
import os, json
import numpy as np

from geniesim.plugins.logger import Logger
from geniesim.utils.system_utils import *
from dataclasses import dataclass
from geniesim.app.controllers.api_core import APICore

logger = Logger()


@dataclass
class RelatedObject:
    asset_path: str
    name: str
    id: str
    category: str
    prim_path: str
    color: str
    size: list
    location_index: int

    def __init__(self, name="", asset_path=""):
        self.id = name
        self.name = name
        self.asset_path = asset_path

    def clarity(self):
        with open(
            os.path.join(assets_path(), self.asset_path, "object_parameters.json"),
            "r",
        ) as f:
            obj_params = json.load(f)
            self.category = obj_params.get("category", "unknown")
            self.color = obj_params.get("color")
            if self.color is None:
                self.color = obj_params.get("llm_descriptions", {}).get("primary_color", "unknown")

        return self

    def update_from_dict(self, data: dict):
        simple_fields = [
            "name",
            "id",
            "category",
            "prim_path",
            "color",
            "size",
            "location_index",
        ]
        for key in simple_fields:
            if key in data:
                setattr(self, key, data[key])


class ObjectSampler(object):
    def __init__(self, api_core: APICore, task_name, instance, assets_path):
        self.api_core = api_core
        self.init_x = -4.5
        self.init_y = 10.65
        self.init_z = 0.85
        self.dx = 0.15
        self.dy = 0.2

        self.task_name = task_name
        self.instance = instance
        self.assets_path = assets_path
        self.object_assets_list = os.listdir(self.assets_path)
        self.rand_low = -0.05
        self.rand_high = 0.05
        self.with_rand = False
        self.generate_table_pose(self.init_x, self.init_y, self.init_z, self.dx, self.dy)

    def generate_table_pose(self, x, y, z, dx, dy):
        self.table_pose = {}
        cnt = 0
        for i in range(3):
            for j in range(1, 4):
                if self.with_rand:
                    randx = np.random.uniform(self.rand_low, self.rand_high)
                    randy = np.random.uniform(self.rand_low, self.rand_high)
                    self.table_pose[cnt] = [x + i * dx + randx, y + j * dy + randy, z]
                else:
                    self.table_pose[cnt] = [x + i * dx, y + j * dy, z]
                cnt += 1

    def get_table_pose(self):
        return self.table_pose

    def sample_assets_path(self):
        # configure seen and unseen object
        with open(
            os.path.join(tgs_conf_path(), "pi_dataset.json"),
            "r",
        ) as f:
            content = json.load(f)

        seen_list = content["seen"]
        unseen_list = content["unseen"]

        n = self.api_core.num_obj
        n_seen = round(n * self.api_core.autogen_ratio)
        sampled_seen = np.random.choice(seen_list, size=n_seen, replace=False)
        if n - n_seen > 0:
            sampled_unseen = np.random.choice(unseen_list, size=n - n_seen, replace=False)
        else:
            sampled_unseen = []
        assets_path_seen = [os.path.join(self.assets_path, item) for item in sampled_seen]
        assets_path_unseen = [os.path.join(self.assets_path, item) for item in sampled_unseen]
        return assets_path_seen + assets_path_unseen

    def generate_scenes_from_input(self):
        logger.info("Generate scenes from input")
        np.random.seed(self.api_core.seed)
        scene_info = {}
        sampled_assets_path = self.sample_assets_path()
        logger.info(f"Add {self.api_core.num_obj} objects")

        pos_index_list = np.random.choice(list(self.table_pose.keys()), size=self.api_core.num_obj, replace=False)

        objects = []
        for idx, path in enumerate(sampled_assets_path):
            obj_id = path.split("/")[-1]
            usda_file_path = os.path.join(path, "Aligned.usd")
            target_prim_path = f"/World/Objects/{obj_id}"

            rotation = [90, 0, 0]
            self.api_core.add_object(usda_file_path, target_prim_path, self.table_pose[pos_index_list[idx]], rotation)

            with open(
                os.path.join(path, "object_parameters.json"),
                "r",
            ) as f:
                obj_params = json.load(f)
                obj = RelatedObject()
                obj.update_from_dict(
                    {
                        "name": obj_id,
                        "id": obj_id,
                        "category": obj_params.get("category", "unknown"),
                        "prim_path": target_prim_path,
                        "color": obj_params.get("color", "unknown"),
                        "location_index": int(pos_index_list[idx]),
                    }
                )
                objects.append(obj)

        scene_info["objects"] = objects

        return scene_info

    def generate_scenes_from_instance(self):
        scene_info = {}
        grids = self.generate_layout()
        if grids is None:
            return scene_info

        objects = []
        for idx, grid in enumerate(grids):
            if grid is None:
                continue

            delta_z = 0
            for obj in grid:
                if obj is None or not isinstance(obj, RelatedObject):
                    continue

                delta_z += 0.02
                usda_file_path = os.path.join(assets_path(), obj.asset_path, "Aligned.usd")
                target_prim_path = f"/World/Objects/{obj.name}_{idx}"
                rotation = [90, 0, 0]
                self.table_pose[idx][2] += delta_z
                self.api_core.add_object(usda_file_path, target_prim_path, self.table_pose[idx], rotation)
                obj.update_from_dict(
                    {
                        "prim_path": target_prim_path,
                        "location_index": int(idx),
                    }
                )
                objects.append(obj)
        scene_info["objects"] = objects
        return scene_info

    def color_layout(self, grid, cfg, indices=[]):
        random_num = cfg.get("random_num", 1)
        assets_root = os.environ.get("SIM_ASSETS")
        assets_path = cfg.get("assets_path")
        existed_color = []
        if indices == []:
            indices = [i for i, v in enumerate(grid) if v is None]

        indices = np.random.choice(indices, size=random_num, replace=False)
        if not assets_root or not assets_path:
            return
        abs_assets_path = os.path.join(assets_root, assets_path)
        if not os.path.exists(abs_assets_path):
            return
        idx = 0
        for p in os.listdir(abs_assets_path):
            if "benchmark_building_blocks_000" in p:
                continue
            abs_p = os.path.join(abs_assets_path, p)
            obj = RelatedObject(p, abs_p).clarity()
            if obj.color != "unknown" and obj.color not in existed_color:
                existed_color.append(obj.color)
                grid[indices[idx]] = [obj]
                idx += 1
            if len(existed_color) >= random_num:
                break

    def random_layout(self, grid, cfg, indices=[]):
        RANDOM_CANDIDATES = cfg["candidates"]
        random_num = cfg.get("random_num", 1)
        remaining_num = grid.count(None)

        if len(RANDOM_CANDIDATES) == 0:
            random_ratio = cfg.get("random_ratio", 0.5)
            if random_num > remaining_num:
                random_num = remaining_num

            if indices == []:
                indices = [i for i, v in enumerate(grid) if v is None]

            with open(
                os.path.join(tgs_conf_path(), "pi_dataset.json"),
                "r",
            ) as f:
                content = json.load(f)

            seen_list = content["seen"]
            unseen_list = content["unseen"]
            n_seen = round(random_num * random_ratio)
            sampled_seen = np.random.choice(seen_list, size=n_seen, replace=False)
            if random_num - n_seen > 0:
                sampled_unseen = np.random.choice(unseen_list, size=random_num - n_seen, replace=False)
            else:
                sampled_unseen = []
            assets_path_seen = [
                {"name": item, "asset_path": os.path.join(self.assets_path, item)} for item in sampled_seen
            ]
            assets_path_unseen = [
                {"name": item, "asset_path": os.path.join(self.assets_path, item)} for item in sampled_unseen
            ]
            grid_index_list = np.random.choice(indices, size=random_num, replace=False)
            for idx, item in enumerate(assets_path_seen + assets_path_unseen):
                if len(grid_index_list) == 0:
                    break
                grid[grid_index_list[idx]] = [RelatedObject(item["name"], item["asset_path"]).clarity()]

            return

        if indices == []:
            indices = [i for i, v in enumerate(grid) if v is None]

        if len(RANDOM_CANDIDATES) == 1:
            grid_index_list = np.random.choice(indices, size=random_num, replace=False)
            for idx in grid_index_list:
                grid[idx] = [RelatedObject(RANDOM_CANDIDATES[0]["name"], RANDOM_CANDIDATES[0]["asset_path"]).clarity()]
        else:
            grid_index_list = np.random.choice(indices, size=random_num, replace=False)
            candidate = np.random.choice(RANDOM_CANDIDATES, size=random_num, replace=False)
            for idx, item in enumerate(grid_index_list):
                choice = candidate[idx]
                grid[item] = [RelatedObject(choice["name"], choice["asset_path"]).clarity()]

    # layout index
    # 8 7 6
    # 5 4 3
    # 2 1 0
    def generate_layout(self):

        problem_file = os.path.join(benchmark_task_definitions_path(), self.task_name, f"problem{self.instance}.json")
        default_problem_file = os.path.join(benchmark_task_definitions_path(), "default_problem.json")

        if os.path.exists(problem_file):
            with open(problem_file, "r") as f:
                content = json.load(f)
        else:
            with open(default_problem_file, "r") as f:
                content = json.load(f)

        if "Init" in content and "Layout" in content["Init"]:
            layout = content["Init"]["Layout"]
        else:
            return None

        grid = [None] * 9
        seed = layout.get("random_seed", 0)
        np.random.seed(seed)  # Ensure reproducibility of random operations
        # 1) Handle specified placements
        for k, v in layout["lay"].items():
            if k == "rest":
                continue

            indices = list(map(int, k.split(",")))
            if v["type"] == "fixed":
                for idx in indices:
                    if isinstance(v["obj"], list):
                        grid[idx] = []
                        for obj in v["obj"]:
                            grid[idx].append(RelatedObject(obj["name"], obj["asset_path"]).clarity())
                    else:
                        grid[idx] = [RelatedObject(v["obj"]["name"], v["obj"]["asset_path"]).clarity()]
            elif v["type"] == "random":
                self.random_layout(grid, rest_cfg, indices=indices)
            elif v["type"] == "empty":
                for idx in indices:
                    grid[idx] = {"object": None, "asset_path": None}
            elif v["type"] == "repeat":
                for idx in indices:
                    grid[idx] = [RelatedObject(v["obj"]["name"], v["obj"]["asset_path"]).clarity()]

        # 2) Handle remaining placements
        if not "rest" in layout["lay"]:
            return grid

        rest_cfg = layout["lay"]["rest"]
        if rest_cfg["type"] == "empty":
            return grid
        elif rest_cfg["type"] == "fixed":
            pass
        elif rest_cfg["type"] == "random":
            self.random_layout(grid, rest_cfg)
        elif rest_cfg["type"] == "color":
            self.color_layout(grid, rest_cfg)

        return grid
