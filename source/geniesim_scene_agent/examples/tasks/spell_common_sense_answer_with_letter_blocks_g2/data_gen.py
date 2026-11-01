# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random
import itertools

def execute_task(robot: RobotAPI):
    # Reduced word list.
    WORD_LIST = [
        "dog", "cat", "pig", "cow", "fox", "bird", "fish", "bear",
        "lion", "wolf", "frog", "duck", "bat", "owl", "crab", "bug",
        "sun", "sky", "sea", "ice", "rock", "sand", "wind", "rain",
        "coal", "fire", "leaf", "cave", "bus", "car", "pen", "cup",
        "box", "hat", "map", "toy", "bed", "key", "nut", "oil",
        "tea", "tie", "gold", "boat", "bowl", "coat", "coin", "desk",
        "dish", "fork", "gift", "kite", "net", "leg", "ear", "toe",
        "hand", "bone", "hair", "run", "fly", "eat", "cry", "play",
        "read", "jump", "old", "hot", "cold", "fast", "red", "blue",
        "pink", "time", "song", "bank", "camp", "cash", "chef", "king"
    ]

    # Lookup table of common-sense questions.
    QUESTION_DICT = [
        {"word": "dog", "question": "什么动物被称为人类最好的朋友，并且会汪汪叫？"},
        {"word": "cat", "question": "什么宠物喜欢抓老鼠，并且发出「喵喵」的叫声？"},
        {"word": "pig", "question": "鼻子圆圆的，身体胖胖的，喜欢在泥巴里打滚的家畜是什么？"},
        {"word": "cow", "question": "我们平时喝的天然牛奶，主要是什么动物产的？"},
        {"word": "fox", "question": "在童话故事中，经常被描绘成非常聪明、狡猾的犬科动物是什么？"},
        {"word": "bird", "question": "长着羽毛和翅膀，通常能够在天上飞翔的动物统称是什么？"},
        {"word": "fish", "question": "生活在水里，用鳃呼吸、用鳍游泳的动物统称是什么？"},
        {"word": "bear", "question": "体型庞大，冬天会躲在洞里冬眠，并且喜欢吃蜂蜜的动物是什么？"},
        {"word": "lion", "question": "雄性脖子上有浓密的鬃毛，被称为「草原之王」的猫科动物是什么？"},
        {"word": "wolf", "question": "狗的野生祖先，喜欢在月圆之夜群聚嚎叫的动物是什么？"},
        {"word": "frog", "question": "小时候是水里的蝌蚪，长大后能在荷叶上蹦跳并呱呱叫的两栖动物是什么？"},
        {"word": "duck", "question": "嘴巴扁扁的，脚上有蹼，游泳时会发出「嘎嘎」叫声的水禽是什么？"},
        {"word": "bat", "question": "唯一能够真正飞行的哺乳动物，经常在夜间出没并倒挂在山洞里的是什么？"},
        {"word": "owl", "question": "眼睛很大，喜欢在夜间活动，叫声像「咕咕」的鸟类是什么？"},
        {"word": "crab", "question": "生活在水里或沙滩上，有两只大钳子，并且总是横着走路的甲壳类动物是什么？"},
        {"word": "bug", "question": "蚂蚁、甲壳虫、毛毛虫等小昆虫的通俗英文叫法是什么？"},
        {"word": "sun", "question": "白天照亮地球，给人类带来光和热的巨大恒星叫什么？"},
        {"word": "sky", "question": "我们白天抬起头，看到的一望无际的蓝色空间叫什么？"},
        {"word": "sea", "question": "面积非常广阔，由咸水组成，连接各大洲的水域叫什么？"},
        {"word": "ice", "question": "水在零度以下的环境中结冰后，形成的固体状态叫什么？"},
        {"word": "rock", "question": "在大山里和河床边随处可见的，非常坚硬的石头叫什么？"},
        {"word": "sand", "question": "去海滩玩耍时，踩在脚底下的细小岩石颗粒叫什么？"},
        {"word": "wind", "question": "树叶被吹动、风筝能飞上天，是因为空气流动产生了什么？"},
        {"word": "rain", "question": "天空中的云朵变成水滴，降落到地面上的天气现象叫什么？"},
        {"word": "coal", "question": "埋在地下，黑色的、可以用来燃烧取暖或发电的化石燃料是什么？"},
        {"word": "fire", "question": "发生火灾时，消防员需要用水去扑灭的红色燃烧物是什么？"},
        {"word": "leaf", "question": "秋天会变黄并从树上掉落，植物用来进行光合作用的部位是什么？"},
        {"word": "cave", "question": "原始人曾经居住过的，在山体或地下形成的天然洞穴叫什么？"},
        {"word": "bus", "question": "体型很大，可以搭载几十名乘客并有固定路线的公共交通工具是什么？"},
        {"word": "car", "question": "有四个轮子，最常见的家用代步机动车是什么？"},
        {"word": "pen", "question": "我们平时用来写字，里面装有墨水或笔芯的文具是什么？"},
        {"word": "cup", "question": "我们平时用来喝水、喝茶或喝咖啡的圆柱形容器叫什么？"},
        {"word": "box", "question": "通常用纸板或木头做成，用来装东西的方形容器是什么？"},
        {"word": "hat", "question": "夏天为了遮阳，冬天为了保暖，人们通常戴在头上的物品是什么？"},
        {"word": "map", "question": "去陌生的地方旅行时，用来查看地理位置和路线规划的图纸是什么？"},
        {"word": "toy", "question": "洋娃娃、积木、遥控车等小孩子最喜欢玩的东西统称为什么？"},
        {"word": "bed", "question": "卧室里必备的，我们每天晚上躺在上面睡觉的家具是什么？"},
        {"word": "key", "question": "出门时必须带的，用来打开防盗门锁的金属小工具是什么？"},
        {"word": "nut", "question": "核桃、杏仁、花生等带有坚硬外壳的干果统称是什么？"},
        {"word": "oil", "question": "炒菜时锅里必须要倒的液体，或者汽车发动机需要的燃料通常叫什么？"},
        {"word": "tea", "question": "用植物的嫩叶泡出来的，中国人和英国人都非常喜欢喝的传统热饮是什么？"},
        {"word": "tie", "question": "男士穿西装打衬衫时，系在脖子上的长条状装饰物是什么？"},
        {"word": "gold", "question": "一种非常珍贵的黄色贵金属，常用来制作昂贵的首饰和奥运会冠军奖牌的是什么？"},
        {"word": "boat", "question": "在湖面或河面上行驶的，通常比较小的水上交通工具是什么？"},
        {"word": "bowl", "question": "中国人吃饭时，用来盛米饭或者装汤的半圆形器皿叫什么？"},
        {"word": "coat", "question": "冬天出门时，为了御寒穿在衣服最外面的长外套或大衣叫什么？"},
        {"word": "coin", "question": "用金属铸造的，圆圆的硬币叫什么？"},
        {"word": "desk", "question": "学生在学校的教室里用来写字和放书本的桌子叫什么？"},
        {"word": "dish", "question": "餐厅里盛菜的盘子，或者做好的「一道菜」通常叫什么？"},
        {"word": "fork", "question": "吃西餐时，和刀子配合使用，带有几个尖齿的餐具是什么？"},
        {"word": "gift", "question": "过生日或过圣诞节时，收到的包装精美的礼物叫什么？"},
        {"word": "kite", "question": "春天起风时，孩子们喜欢放飞到天上，由一根线牵引的玩具是什么？"},
        {"word": "net", "question": "打网球或羽毛球时拦在场地中间的网，或者渔民用来捕鱼的网叫什么？"},
        {"word": "leg", "question": "人体用来站立、行走和奔跑的下肢部位叫什么？"},
        {"word": "ear", "question": "长在头部两侧，人类和动物用来听声音的器官是什么？"},
        {"word": "toe", "question": "长在脚的前端，正常人每只脚上都有五个的部位叫什么？"},
        {"word": "hand", "question": "人体手臂的前端，长着五根手指，用来拿东西的部位是什么？"},
        {"word": "bone", "question": "支撑人体和动物身体的坚硬白色骨架部分叫什么？"},
        {"word": "hair", "question": "长在人的头顶上，可以去理发店修剪和做造型的毛发叫什么？"},
        {"word": "run", "question": "参加马拉松比赛时，运动员所做的一项比走路快的运动是什么？"},
        {"word": "fly", "question": "鸟类、蝙蝠和飞机在天空中移动的方式叫什么？"},
        {"word": "eat", "question": "感到饥饿时，我们需要把食物放进嘴里咀嚼并吞下的动作叫什么？"},
        {"word": "cry", "question": "感到非常伤心或身体疼痛时，眼睛里流出眼泪的动作叫什么？"},
        {"word": "play", "question": "小朋友们在操场上做游戏、玩耍的动作叫什么？"},
        {"word": "read", "question": "翻开书本，看着上面的文字并理解其中意思的动作叫什么？"},
        {"word": "jump", "question": "双脚离开地面，身体向上跃起的动作叫什么？"},
        {"word": "old", "question": "和崭新（new）意思相反，形容一个东西用了很久的词是什么？"},
        {"word": "hot", "question": "炎热的夏天，或者刚刚烧开的开水，给人的感觉是什么？"},
        {"word": "cold", "question": "寒冷的冬天去室外，人们通常会感到的温度状态是什么？"},
        {"word": "fast", "question": "猎豹奔跑的速度，或者跑车行驶的速度可以用哪个词来形容？"},
        {"word": "red", "question": "交通十字路口的信号灯中，代表停止的颜色是什么？"},
        {"word": "blue", "question": "在晴朗的白天，没有云朵的天空通常是什么颜色的？"},
        {"word": "pink", "question": "经常被用来代表浪漫，由红色和白色混合而成的颜色是什么？"},
        {"word": "time", "question": "墙上的钟表和手腕上的手表，是用来告诉我们什么在流逝的？"},
        {"word": "song", "question": "歌手在舞台上用优美的旋律演唱出来的音乐作品叫什么？"},
        {"word": "bank", "question": "我们平时用来存钱、取钱或者办理贷款的金融机构叫什么？"},
        {"word": "camp", "question": "带着帐篷去野外生火、露营过夜的户外活动叫什么？"},
        {"word": "cash", "question": "纸币和硬币等可以直接用来实体交易的现金统称为什么？"},
        {"word": "chef", "question": "在高级餐厅的后厨里，负责烹饪美味佳肴的厨师长叫什么？"},
        {"word": "king", "question": "在古代的王国里，拥有最高权力的男性君主叫什么？"}
    ]

    # Convert words to uppercase and remove duplicates.
    WORD_SET = set([word.upper() for word in WORD_LIST])

    # Build the word-to-question mapping.
    WORD_TO_QUESTION = {item["word"].upper(): item["question"] for item in QUESTION_DICT}

    # Initialize letter-asset information.
    objects_list = {
        "benchmark_letter_026": {"has_interactions": True, "object_type": "letter_block", "letter": "A"},
        "benchmark_letter_027": {"has_interactions": True, "object_type": "letter_block", "letter": "B"},
        "benchmark_letter_028": {"has_interactions": True, "object_type": "letter_block", "letter": "C"},
        "benchmark_letter_029": {"has_interactions": True, "object_type": "letter_block", "letter": "D"},
        "benchmark_letter_030": {"has_interactions": True, "object_type": "letter_block", "letter": "E"},
        "benchmark_letter_031": {"has_interactions": True, "object_type": "letter_block", "letter": "F"},
        "benchmark_letter_032": {"has_interactions": True, "object_type": "letter_block", "letter": "G"},
        "benchmark_letter_033": {"has_interactions": True, "object_type": "letter_block", "letter": "H"},
        "benchmark_letter_034": {"has_interactions": True, "object_type": "letter_block", "letter": "I"},
        "benchmark_letter_035": {"has_interactions": True, "object_type": "letter_block", "letter": "J"},
        "benchmark_letter_036": {"has_interactions": True, "object_type": "letter_block", "letter": "K"},
        "benchmark_letter_037": {"has_interactions": True, "object_type": "letter_block", "letter": "L"},
        "benchmark_letter_038": {"has_interactions": True, "object_type": "letter_block", "letter": "M"},
        "benchmark_letter_039": {"has_interactions": True, "object_type": "letter_block", "letter": "N"},
        "benchmark_letter_040": {"has_interactions": True, "object_type": "letter_block", "letter": "O"},
        "benchmark_letter_041": {"has_interactions": True, "object_type": "letter_block", "letter": "P"},
        "benchmark_letter_042": {"has_interactions": True, "object_type": "letter_block", "letter": "Q"},
        "benchmark_letter_043": {"has_interactions": True, "object_type": "letter_block", "letter": "R"},
        "benchmark_letter_044": {"has_interactions": True, "object_type": "letter_block", "letter": "S"},
        "benchmark_letter_045": {"has_interactions": True, "object_type": "letter_block", "letter": "T"},
        "benchmark_letter_046": {"has_interactions": True, "object_type": "letter_block", "letter": "U"},
        "benchmark_letter_047": {"has_interactions": True, "object_type": "letter_block", "letter": "V"},
        "benchmark_letter_048": {"has_interactions": True, "object_type": "letter_block", "letter": "W"},
        "benchmark_letter_049": {"has_interactions": True, "object_type": "letter_block", "letter": "X"},
        "benchmark_letter_050": {"has_interactions": True, "object_type": "letter_block", "letter": "Y"},
        "benchmark_letter_051": {"has_interactions": True, "object_type": "letter_block", "letter": "Z"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    # 1. Find the table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find all letter blocks.
    all_letter_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")

        # Check whether the object is a letter block.
        is_letter_block = False
        letter_value = None
        for asset_id, asset_info in objects_list.items():
            if asset_id in usd and asset_info["object_type"] == "letter_block":
                is_letter_block = True
                letter_value = asset_info["letter"]
                break

        if is_letter_block:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)
            if pos is not None and size is not None:
                all_letter_blocks.append({
                    "id": obj_id,
                    "letter": letter_value,
                    "position": pos,
                    "rotation": rot,
                    "size": size,
                })
                print(f"Found letter block: {obj_id}, letter: {letter_value}")

    if len(all_letter_blocks) < 3:
        print(f"Found only {len(all_letter_blocks)} letter blocks, need at least 3")
        return False

    print(f"Total letter blocks found: {len(all_letter_blocks)}")

    # 3. Analyze all words that can be formed from the scene's letters.
    # Given N letter blocks, find every word that can be formed.

    available_letters = [block["letter"] for block in all_letter_blocks]
    print(f"Available letters in scene: {available_letters}")

    # Build a letter-to-block mapping for lookup.
    letter_to_blocks = {}
    for block in all_letter_blocks:
        letter = block["letter"]
        if letter not in letter_to_blocks:
            letter_to_blocks[letter] = []
        letter_to_blocks[letter].append(block)

    # Find all possible words.
    possible_words = []
    word_to_blocks_mapping = {}  # Store the blocks for each word.
    word_to_question_mapping = {}  # Store the question for each word.

    print("\nSearching for all possible words from available letters...")

    # The set of letters in the scene.
    available_letters_set = set(available_letters)

    # Iterate over every word in the vocabulary.
    for word in WORD_SET:
        word_letters_set = set(word)

        # Check that the scene contains every letter of the word.
        # Use a subset check because letters are not repeated.
        if word_letters_set.issubset(available_letters_set):
            # Verify there are enough letters, without repeated letters.
            if len(word) <= len(available_letters):
                # Find a block for each letter of the word.
                letter_to_blocks_copy = {}
                for block in all_letter_blocks:
                    letter = block["letter"]
                    if letter not in letter_to_blocks_copy:
                        letter_to_blocks_copy[letter] = []
                    letter_to_blocks_copy[letter].append(block)

                blocks_for_word = []
                can_spell = True

                for letter in word:
                    if letter in letter_to_blocks_copy and letter_to_blocks_copy[letter]:
                        # Take the first available block.
                        block = letter_to_blocks_copy[letter].pop(0)
                        blocks_for_word.append(block)
                    else:
                        can_spell = False
                        break

                if can_spell and len(blocks_for_word) == len(word):
                    possible_words.append(word)
                    word_to_blocks_mapping[word] = blocks_for_word
                    word_to_question_mapping[word] = WORD_TO_QUESTION.get(word, f"拼写单词 {word.lower()}")
                    print(f"  Found possible word: {word} (question: {WORD_TO_QUESTION.get(word, 'no question')})")

    if not possible_words:
        print("ERROR: No possible words found from available letters")
        print(f"Available letters: {available_letters}")
        return False

    print(f"\nFound {len(possible_words)} possible words:")
    for i, word in enumerate(possible_words):
        print(f"  {i+1}. {word} - {WORD_TO_QUESTION.get(word, 'spell the word')}")

    # 4. Randomly choose among the possible words.
    target_word = random.choice(possible_words)
    target_question = word_to_question_mapping[target_word]
    found_blocks_for_word = word_to_blocks_mapping[target_word]

    print(f"\n=== Randomly selected word to spell: {target_word} ===")
    print(f"Corresponding question: {target_question}")
    print(f"Selected blocks for spelling (in word order):")
    for i, block in enumerate(found_blocks_for_word):
        print(f"  {i+1}. Letter {block['letter']} (id: {block['id']})")

    # 5. MANDATORY: set the task description.
    task_description = {
        "task_name": f"based on the question '{target_question}', spell the answer word '{target_word.lower()}' with letter blocks",
        "english_task_name": f"based on the question '{target_question}', spell the answer word '{target_word.lower()}' with letter blocks",
        "init_scene_text": f"The robot is in front of the table, there are letter blocks on the table, and there's a question to answer"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 6. Compute targets using the same placement logic as the reference script.
    # Target X range: (-0.05, 0.0); sample an X coordinate.
    target_x = np.random.uniform(-0.05, 0.0)
    print(f"Randomly selected target_x: {target_x:.3f}")

    # Starting Y range: (0.05, 0.08); sample a value.
    start_y = np.random.uniform(0.05, 0.08)
    print(f"Randomly selected start_y: {start_y:.3f}")

    # Block spacing: 0.07 m (7 cm).
    spacing = 0.07

    # Compute left-to-right targets with decreasing Y.
    target_positions = []
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0
    margin = 0.01  # Safety clearance.

    for i in range(len(found_blocks_for_word)):
        target_y = start_y - spacing * i
        block_size = found_blocks_for_word[i]["size"]
        target_z = table_top_z + margin + block_size[2] * 3 / 2.0

        target_position = np.array([target_x, target_y, target_z])
        target_positions.append(target_position)

        letter = found_blocks_for_word[i]["letter"]
        print(f"Target position for letter {letter}: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")

    # 7. Preserve the initial orientation during placement.
    def calculate_target_rotation(initial_rotation):
        return quat2mat_wxyz(initial_rotation)

    # 8. Place the blocks in the word's letter order.
    placed_blocks = []
    last_arm = ""

    for i, block in enumerate(found_blocks_for_word):
        block_id = block["id"]
        block_letter = block["letter"]
        current_pos = block["position"]
        current_rot = block["rotation"]

        print(f"\n=== Processing letter {i+1}/{len(found_blocks_for_word)}: {block_letter} ===")

        arm = robot.judge_arm_to_use(current_pos)
        print(f"Using arm: {arm}")

        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        # List of immovable objects.
        unmovable_objects = [table_id] + [b["id"] for b in placed_blocks]

        # Grasp configuration.
        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects,
            action_description={
                "zh": f"拿起字母{block_letter}",
                "en": f"picks up the letter {block_letter}",
            },
        )

        # Target rotation matrix.
        target_rot_matrix = calculate_target_rotation(current_rot)

        # Generate multiple candidate poses.
        target_poses = []
        base_position = target_positions[i]

        x_offsets = np.linspace(-0.002, 0.002, 3)
        y_offsets = np.linspace(-0.002, 0.002, 3)
        z_offsets = np.linspace(0, 0.003, 3)

        for x_off in x_offsets:
            for y_off in y_offsets:
                for z_off in z_offsets:
                    position = base_position + np.array([x_off, y_off, z_off])
                    pose = np.eye(4)
                    pose[:3, :3] = target_rot_matrix
                    pose[:3, 3] = position
                    target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for letter {block_letter}")

        # Create the placement checker.
        def create_placement_checker(block_id, expected_position, block_letter):
            def check_placement():
                current_p, current_r = robot.get_object_pose(block_id)
                if current_p is None or current_r is None:
                    return False
                xy_dist = np.linalg.norm(current_p[:2] - expected_position[:2])
                if xy_dist > 0.03:
                    print(f"Letter {block_letter} position error: xy_dist={xy_dist:.3f}m")
                    return False
                return True
            return check_placement

        # Placement description.
        if len(found_blocks_for_word) == 3:
            position_desc = "最左侧" if i == 0 else ("中间" if i == 1 else "最右侧")
        else:
            position_desc = "最左侧" if i == 0 else ("最右侧" if i == len(found_blocks_for_word)-1 else "中间")

        place_desc_zh = f"将字母{block_letter}放置到{position_desc}位置"
        place_desc_en = f"places the letter {block_letter} at the {position_desc} position"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[create_placement_checker(block_id, base_position, block_letter)],
            action_description={"zh": place_desc_zh, "en": place_desc_en},
            disable_upside_down=False,
        )

        # Execute grasping and placement.
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config,
        )

        if not current_success:
            print(f"Failed to place letter {block_letter}")
            return False

        print(f"Successfully placed letter {block_letter}")
        placed_blocks.append(block)

    # 9. Final validation.
    print("\n=== Performing final verification ===")

    if len(placed_blocks) == len(found_blocks_for_word):
        # Collect final positions.
        final_positions = []
        for block in placed_blocks:
            pos, _ = robot.get_object_pose(block["id"])
            if pos is not None:
                final_positions.append((block["letter"], pos[1], block["id"]))

        # Sort by decreasing Y, from left to right.
        final_positions.sort(key=lambda x: x[1], reverse=True)

        print("Final arrangement from left to right:")
        letters_in_order = []
        for letter, y_pos, block_id in final_positions:
            print(f"  Letter {letter} at y={y_pos:.3f}")
            letters_in_order.append(letter)

        # Build the string spelled by the placed blocks.
        spelled_word = ''.join(letters_in_order)

        # Check that the spelled word belongs to the vocabulary.
        if spelled_word.upper() in WORD_SET:
            print(f"✅ SUCCESS: Successfully answered the question!")
            print(f"  Question: {target_question}")
            print(f"  Answer: {spelled_word}")
            print(f"  Letters in order: {' '.join(letters_in_order)}")

            # Check the spacing.
            print("\nChecking spacing between letters:")
            for j in range(len(final_positions) - 1):
                letter1, y1, _ = final_positions[j]
                letter2, y2, _ = final_positions[j + 1]
                actual_spacing = abs(y1 - y2)
                if abs(actual_spacing - spacing) > 0.02:
                    print(f"  Warning: Spacing between {letter1} and {letter2} is {actual_spacing:.3f}m")
                else:
                    print(f"  Good: Spacing between {letter1} and {letter2} is {actual_spacing:.3f}m")
        else:
            print(f"❌ ERROR: Spelled word '{spelled_word}' is not the correct answer")
            print(f"  Question: {target_question}")
            print(f"  Expected answer: {target_word}")
            print(f"  Letters in order: {' '.join(letters_in_order)}")
            success = False
    else:
        print(f"❌ ERROR: Only placed {len(placed_blocks)} out of {len(found_blocks_for_word)} letters")
        success = False

    # Return the arm to its origin.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    return success and (len(placed_blocks) == len(found_blocks_for_word))
