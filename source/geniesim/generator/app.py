#!/usr/bin/env python3


# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
from networkx.readwrite import json_graph

from scipy.spatial.transform import Rotation as R
from pprint import pprint

######
from helper import *


import geniesim.generator.scene_language.mi_helper
import shutil


current_path = os.path.abspath(__file__)
GENIESIM_PATH = os.path.dirname(os.path.dirname(current_path))


def copy_llm_result(save_path):
    llm_result_path = os.path.join(os.path.dirname(current_path), "LLM_RESULT.py")
    shutil.copy(llm_result_path, save_path)


def main(args):
    task_template_path = args.template_path
    if task_template_path != "" and os.path.exists(task_template_path):
        shutil.copy(task_template_path, os.path.join(GENIESIM_PATH, "generator/LLM_RESULT.py"))

    from LLM_RESULT import root_scene

    # scene gen
    scene_data = root_scene()
    scene_info, G = gen_scene_layout_info(scene_data)
    # pprint(scene_info)

    # output folder
    if args.scene_id != "":
        scene_path0_dir = os.path.join(GENIESIM_PATH, f"benchmark/config/llm_task/{args.scene_id}")
    else:
        scene_path0_dir = os.path.join(GENIESIM_PATH, f"benchmark/config/llm_task/{scene_info['scene_id']}")
    if not os.path.exists(scene_path0_dir):
        os.makedirs(scene_path0_dir, exist_ok=True)
    num = sum([len(d) for r, d, folder in os.walk(scene_path0_dir)])
    scene_path1_dir = os.path.join(str(scene_path0_dir), f"{num}")
    os.makedirs(os.path.join(str(scene_path0_dir), f"{num}"), exist_ok=True)

    # dump info
    nx.nx_agraph.write_dot(G, os.path.join(scene_path1_dir, "graph.dot"))
    H = nx.nx_agraph.to_agraph(G)
    for n in G.nodes:
        tags = G.nodes[n].get("tags", [])
        H.get_node(n).attr["tags"] = f"{n}\\n{','.join(tags)}"
    H.draw(os.path.join(scene_path1_dir, "graph.svg"), prog="dot")
    with open(scene_path1_dir + "/scene_info.json", "w") as f:
        json.dump(scene_info, f, indent=2)
    with open(scene_path1_dir + "/scene_info.json", "r") as f:
        scene_info_load = json.load(f)

    # scene path
    scene_path = scene_path1_dir + "/scene.usda"

    from geniesim.generator.utils.usd import gen_scene_usda

    object_info_list = []
    for key, val in scene_info["layout"].items():
        usd_name = val["usd"]
        url = ASSETS_INDEX[usd_name]["url"]
        type = usd_name.split("_")[0]
        translation = val["xyz"]
        rotation = val["xyzw"]
        scale = [1, 1, 1]
        object_info_list.append(
            {
                "id": key,
                "usd": usd_name,
                "url": url,
                "type": type,
                "translation": translation,
                "rotation": rotation,
                "scale": scale,
            }
        )

    # convert to usda
    gen_scene_usda(scene_path, object_info_list)

    # fmt: off
    copy_llm_result(os.path.join(scene_path1_dir, "LLM_RESULT.py"))

    # fmt: on

    # API gallery
    print("Scene Graph DAG is Here!")
    # assert nx.is_directed_acyclic_graph(GG)
    GG: nx.DiGraph = nx.json_graph.node_link_graph(scene_info["relations"]["graph"])

    if args.task_gen:
        from geniesim.evaluator.generators.instruction_gen import generate_instructions
        from geniesim.evaluator.generators.eval_gen import generate_problems

        print("\n\nGenerating instructions...")

        generate_instructions(scene_dir=scene_path1_dir, instruction_category=args.scene_id)

        print("\n\nGenerating problems...")
        generate_problems(scene_path1_dir)

    exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate instructions and problems")
    parser.add_argument("--scene_id", type=str, default="", help="scene_id to save")
    parser.add_argument("--task_gen", action="store_true", default=False, help="Generate Task")
    parser.add_argument("--template_path", type=str, default="", help="LLM_RESULT template path")
    args = parser.parse_args()
    main(args)
