# 🧠 Benchmark Evaluation Framework

Use ADER (Action Domain Evaluation Rule) for evaluation configuration

---

## 📦 What is ader?

---

## 🚀 Features

| Feature | Description | Status |
|----------|--------------|---------|
| Evaluation Configuration | Define and manage structured evaluation settings | ✅ Stable |
| Rule-Based Evaluation | Apply customizable rules to evaluate system outputs | ✅ Stable |
| Action-Domain Abstraction | Organize evaluation logic by actions and domains | ✅ Stable |
| YAML/JSON Support | Load configurations from human-readable formats | ✅ Stable |
| Extensible Plugin System | Add new evaluation rules dynamically | 🚧 In Progress |

---

## ⚙️ Example Configuration
```json
{
  "Acts": [
    {
      "ActionList": [
        {
          "ActionSetWaitAny": [
            {
              "Follow": "beverage_bottle_002|[0.2,0.2,0.2]|right"
            },
            {
              "Onfloor": "beverage_bottle_002|0.0"
            }
          ]
        },
        {
          "ActionSetWaitAny": [
            {
              "PickUpOnGripper": "beverage_bottle_002|right"
            },
            {
              "Timeout": 120
            },
            {
              "Onfloor": "beverage_bottle_002|0.0"
            }
          ]
        },
        {
          "ActionSetWaitAny": [
            {
              "Inside": "beverage_bottle_002|handbag_000|1"
            },
            {
              "StepOut": 1000
            }
          ]
        }
      ]
    }
  ],
  "Init": [],
  "Objects": [],
  "Problem": "pack_in_the_supermarket"
}
```

---

## 🧩 Current Evaluation Capability

| Action | Description | Base Class | Syntax |
|--------|------------|------------|--------|
| **COMMON** | | | |
| ActionList | Queue action: Internal actions execute sequentially. | ActionBase | `"ActionList":[]` |
| ActionSetWaitAny | Conditional queue action: Completes when any internal action is done. | ActionBase | `"ActionSetWaitAny":[]` |
| ActionWaitForTime | Time wait action: Similar to sleep, but does not block the thread. | ActionBase | `"ActionWaitForTime": 3.0` |
| TimeOut | Timeout validation action: Checks if a timeout has occurred. | ActionCancelBase | `"Timeout": 60` |
| StepOut | Step limit validation action: Checks if the step count limit has been reached. | ActionCancelBase | `"StepOut": 100` |
| ActionSetWaitAll | Exit when all conditions are met. | ActionBase | `"ActionSetWaitAll":[]` |
| **CUSTOM** | | | |
| Ontop | An object is above another object | EvaluateAction | `"Ontop": "active_obj|passive_obj"` |
| Inside | An object is inside another object | EvaluateAction | `"Inside": "active_obj|passive_obj|scale_factor"` |
| PushPull | Check if the sliding joint of an articulated object is within threshold [min, max] — used to determine if a drawer-like object is open or closed. | EvaluateAction | `"PushPull": "obj_id|thresh_min|thresh_max"` |
| Follow | Check if the left/right gripper is following a specific object, within a bounding box-defined range [x, y, z]. | EvaluateAction | `"Follow": "obj_id|bbox|gripper_id"` |
| PickUpOnGripper | Gripper grasps an object | EvaluateAction | `"PickUpOnRightGripper": "object|gripper_id"` |
| OnShelf | The object is inside a specific region | EvaluateAction | `"OnShelf": "obj_id|target_id|bbox|height"` |
| Onfloor | Check if a specified object has fallen below a reference height `ref_z`; if so, exit | ActionCancelBase | `"Inside": "obj_id|ref_z"` |
| Cover | Object A covers object B | EvaluateAction | `"Cover": "active_obj|passive_obj"` |
