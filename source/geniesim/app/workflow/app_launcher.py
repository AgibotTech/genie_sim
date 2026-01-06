# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Sub-package with the utility class to configure the :class:`isaacsim.kit.SimulationApp`.
"""

import argparse
import contextlib
import os, sys, re
import signal
from typing import Any, Literal

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim

from isaacsim import SimulationApp


class AppLauncher:
    def __init__(self, app_config):

        # Define config members that are read from env-vars or keyword args
        self._headless: bool  # 0: GUI, 1: Headless
        self._livestream: Literal[0, 1, 2]  # 0: Disabled, 1: Native, 2: WebRTC
        self._offscreen_render: bool  # 0: Disabled, 1: Enabled
        self._sim_experience_file: str  # Experience file to load
        self._render_mode = "RealTimePathTracing"

        # Exposed to train scripts
        self.device_id: int  # device ID for GPU simulation (defaults to 0)
        self.local_rank: int  # local rank of GPUs in the current node
        self.global_rank: int  # global rank for multi-node training

        self.device_id: int  # device ID for GPU simulation (defaults to 0)
        self.local_rank: int  # local rank of GPUs in the current node
        self.global_rank: int  # global rank for multi-node training

        self._config_resolution(app_config)
        self._create_app()
        # -- during interrupts
        signal.signal(signal.SIGINT, self._interrupt_signal_handle_callback)
        # -- during explicit `kill` commands
        signal.signal(signal.SIGTERM, self._abort_signal_handle_callback)
        # -- during segfaults
        signal.signal(signal.SIGABRT, self._abort_signal_handle_callback)
        signal.signal(signal.SIGSEGV, self._abort_signal_handle_callback)

    """
    Properties.
    """

    @property
    def app(self) -> SimulationApp:
        """The launched SimulationApp."""
        if self._app is not None:
            return self._app
        else:
            raise RuntimeError("The `AppLauncher.app` member cannot be retrieved until the class is initialized.")

    """
    Operations.
    """

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        parser_help = None
        if len(parser._actions) > 0 and isinstance(parser._actions[0], argparse._HelpAction):
            parser_help = parser._actions[0]
            parser._option_string_actions.pop("-h")
            parser._option_string_actions.pop("--help")

        known, _ = parser.parse_known_args()
        config = vars(known)
        if len(config) == 0:
            logger.warning(
                "[WARN][AppLauncher]: There are no arguments attached to the ArgumentParser object."
                " If you have your own arguments, please load your own arguments before calling the"
                " `AppLauncher.add_app_launcher_args` method. This allows the method to check the validity"
                " of the arguments and perform checks for argument names."
            )
        else:
            AppLauncher._check_argparser_config_params(config)

        arg_group = parser.add_argument_group(
            "app_launcher arguments",
            description="Arguments for the AppLauncher. For more details, please check the documentation.",
        )

        arg_group.add_argument(
            "--headless",
            action="store_true",
            default=False,
        )
        arg_group.add_argument(
            "--livestream",
            type=int,
            default=AppLauncher._APPLAUNCHER_CFG_INFO["livestream"][1],
            choices={0, 1, 2},
            help="Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.",
        )
        arg_group.add_argument(
            "--enable_cameras",
            action="store_true",
            default=AppLauncher._APPLAUNCHER_CFG_INFO["enable_cameras"][1],
            help="Enable camera sensors and relevant extension dependencies.",
        )
        arg_group.add_argument(
            "--device",
            type=str,
            default=AppLauncher._APPLAUNCHER_CFG_INFO["device"][1],
            help='The device to run the simulation on. Can be "cpu", "cuda", "cuda:N", where N is the device ID',
        )
        # Add the deprecated cpu flag to raise an error if it is used
        arg_group.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
        arg_group.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose-level log output from the SimulationApp.",
        )
        arg_group.add_argument(
            "--info",
            action="store_true",
            help="Enable info-level log output from the SimulationApp.",
        )
        arg_group.add_argument(
            "--experience",
            type=str,
            default="",
            help=(
                "The experience file to load when launching the SimulationApp. If an empty string is provided,"
                " the experience file is determined based on the headless flag. If a relative path is provided,"
                " it is resolved relative to the `apps` folder in Isaac Sim and Isaac Lab (in that order)."
            ),
        )
        arg_group.add_argument(
            "--kit_args",
            type=str,
            default="",
            help=(
                "Command line arguments for Omniverse Kit as a string separated by a space delimiter."
                ' Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2"'
            ),
        )

        if parser_help is not None:
            parser._option_string_actions["-h"] = parser_help
            parser._option_string_actions["--help"] = parser_help

    @staticmethod
    def _check_argparser_config_params(config: dict) -> None:
        """
        1. prevent name conflicts
        2. type validation
        """
        applauncher_keys = set(AppLauncher._APPLAUNCHER_CFG_INFO.keys())
        for key, value in config.items():
            if key in applauncher_keys:
                raise ValueError(
                    f"The passed ArgParser object already has the field '{key}'. This field will be added by"
                    " `AppLauncher.add_app_launcher_args()`, and should not be added directly. Please remove the"
                    " argument or rename it to a non-conflicting name."
                )
        # check that type of the passed keys are valid
        simulationapp_keys = set(AppLauncher._SIM_APP_CFG_TYPES.keys())
        for key, value in config.items():
            if key in simulationapp_keys:
                given_type = type(value)
                expected_types = AppLauncher._SIM_APP_CFG_TYPES[key]
                if type(value) not in set(expected_types):
                    raise ValueError(
                        f"Invalid value type for the argument '{key}': {given_type}. Expected one of {expected_types},"
                        " if intended to be ingested by the SimulationApp object. Please change the type if this"
                        " intended for the SimulationApp or change the name of the argument to avoid name conflicts."
                    )
                # log values which will be used
                logger.warning(
                    f"[INFO][AppLauncher]: The argument '{key}' will be used to configure the SimulationApp."
                )

    def _config_resolution(self, app_config):
        self._headless = app_config.headless
        self._render_mode = app_config.render_mode

    def _create_app(self):
        """Launch SimulationApp"""
        self._app = SimulationApp(
            {
                "headless": self._headless,
                "disable_viewport_updates": self._headless,
                "renderer": self._render_mode,
                "limit_cpu_threads": 16,
                "extra_args": [
                    # "--/persistent/rtx/modes/rt2/enabled=true",
                ],
            }
        )

    _APPLAUNCHER_CFG_INFO: dict[str, tuple[list[type], Any]] = {
        "headless": ([bool], False),
        "livestream": ([int], -1),
        "enable_cameras": ([bool], False),
        "device": ([str], "cuda:0"),
        "experience": ([str], ""),
    }

    _SIM_APP_CFG_TYPES: dict[str, list[type]] = {
        "headless": [bool],
        "hide_ui": [bool, type(None)],
        "active_gpu": [int, type(None)],
        "physics_gpu": [int],
        "multi_gpu": [bool],
        "sync_loads": [bool],
        "width": [int],
        "height": [int],
        "window_width": [int],
        "window_height": [int],
        "display_options": [int],
        "subdiv_refinement_level": [int],
        "renderer": [str],
        "anti_aliasing": [int],
        "samples_per_pixel_per_frame": [int],
        "denoiser": [bool],
        "max_bounces": [int],
        "max_specular_transmission_bounces": [int],
        "max_volume_bounces": [int],
        "open_usd": [str, type(None)],
        "livesync_usd": [str, type(None)],
        "fast_shutdown": [bool],
        "experience": [str],
    }

    def _interrupt_signal_handle_callback(self, signal, frame):
        """Handle the interrupt signal from the keyboard."""
        self._app.close()
        raise KeyboardInterrupt

    def _abort_signal_handle_callback(self, signal, frame):
        """Handle the abort/segmentation/kill signals."""
        self._app.close()
