# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Local hold-only CoRobot fixture, for checking transport without model weights."""

import argparse

import msgpack
from websockets.sync.server import serve


def hold_reply(payload, *, stop=False):
    if payload.get("method") != "infer":
        raise ValueError("expected CoRobot infer request")
    if stop:
        return {"result": {"episode_done": True}}
    state = payload["params"]["states"]
    arms, hands = state["arm_joint_states"], state["gripper_states"]
    if len(arms) != 14 or len(hands) != 40:
        raise ValueError("expected Wuji arm14 + hand40 state")
    return {
        "result": {
            "left_arm": {"kind": "JOINT_ABS", "values": [arms[:7]]},
            "right_arm": {"kind": "JOINT_ABS", "values": [arms[7:]]},
            "left_effector": [hands[:20]],
            "right_effector": [hands[20:]],
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--requests", type=int, default=3, help="hold requests before the model-stop fixture")
    options = parser.parse_args(argv)
    if options.requests < 1:
        parser.error("--requests must be positive")

    def handler(connection):
        for index in range(options.requests + 1):
            payload = msgpack.unpackb(connection.recv(), raw=False)
            reply = hold_reply(payload, stop=index == options.requests)
            connection.send(msgpack.packb(reply))

    with serve(handler, options.host, options.port) as server:
        print(f"Hold-only CoRobot fixture at ws://{options.host}:{server.socket.getsockname()[1]}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
