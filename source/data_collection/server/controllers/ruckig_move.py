# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from copy import copy

from ruckig import InputParameter, OutputParameter, Result, Ruckig

from common.base_utils.logger import logger


class RuckigController:
    def __init__(self, dof_num, delta_time):
        self.otg = Ruckig(dof_num, delta_time)
        self.inp = InputParameter(dof_num)
        self.out = OutputParameter(dof_num)
        self.dof_num = dof_num

    def caculate_trajectory(self, current_position, target_position):
        self.inp.current_position = current_position
        self.inp.target_position = target_position
        self.inp.max_velocity = [200] * self.dof_num
        self.inp.max_acceleration = [10] * self.dof_num
        self.inp.max_jerk = [50] * self.dof_num
        first_output, out_list = None, []
        res = Result.Working
        while res == Result.Working:
            res = self.otg.update(self.inp, self.out)
            out_list.append(copy(self.out.new_position))
            self.out.pass_to_input(self.inp)
            if not first_output:
                first_output = copy(self.out)
        return out_list


if __name__ == "__main__":
    # Create instances: the Ruckig OTG as well as input and output parameters
    otg = Ruckig(3, 10)  # DoFs, control cycle
    inp = InputParameter(3)
    out = OutputParameter(3)

    # Set input parameters
    inp.current_position = [0.0, 0.0, 0.5]

    inp.target_position = [5.0, -2.0, -3.5]

    inp.max_velocity = [3.0, 1.0, 3.0]
    inp.max_acceleration = [3.0, 2.0, 1.0]
    inp.max_jerk = [4.0, 3.0, 2.0]

    logger.info("\t".join(["t"] + [str(i) for i in range(otg.degrees_of_freedom)]))

    # Generate the trajectory within the control loop
    first_output, out_list = None, []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)

        logger.info("\t".join([f"{out.time:0.3f}"] + [f"{p:0.3f}" for p in out.new_velocity]))
        out_list.append(copy(out))

        out.pass_to_input(inp)

        if not first_output:
            first_output = copy(out)

    logger.info(f"Calculation duration: {first_output.calculation_duration:0.1f} [µs]")
    logger.info(f"Trajectory duration: {first_output.trajectory.duration:0.4f} [s]")

    # Plot the trajectory
