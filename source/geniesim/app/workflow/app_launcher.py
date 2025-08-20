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

from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim

from isaacsim import SimulationApp


class AppLauncher:
    def __init__(
        self, launcher_args: argparse.Namespace | dict | None = None, **kwargs
    ):
        if launcher_args is None:
            launcher_args = {}
        elif isinstance(launcher_args, argparse.Namespace):
            launcher_args = launcher_args.__dict__

        # Check that arguments are unique
        if len(kwargs) > 0:
            if not set(kwargs.keys()).isdisjoint(launcher_args.keys()):
                overlapping_args = set(kwargs.keys()).intersection(launcher_args.keys())
                raise ValueError(
                    f"Input `launcher_args` and `kwargs` both provided common attributes: {overlapping_args}."
                    " Please ensure that each argument is supplied to only one of them, as the AppLauncher cannot"
                    " discern priority between them."
                )
            launcher_args.update(kwargs)

        # Define config members that are read from env-vars or keyword args
        self._headless: bool  # 0: GUI, 1: Headless
        self._livestream: Literal[0, 1, 2]  # 0: Disabled, 1: Native, 2: WebRTC
        self._offscreen_render: bool  # 0: Disabled, 1: Enabled
        self._sim_experience_file: str  # Experience file to load
        self._render_mode = "RaytracedLighting"

        # Exposed to train scripts
        self.device_id: int  # device ID for GPU simulation (defaults to 0)
        self.local_rank: int  # local rank of GPUs in the current node
        self.global_rank: int  # global rank for multi-node training

        self.device_id: int  # device ID for GPU simulation (defaults to 0)
        self.local_rank: int  # local rank of GPUs in the current node
        self.global_rank: int  # global rank for multi-node training

        self._config_resolution(launcher_args)
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
            raise RuntimeError(
                "The `AppLauncher.app` member cannot be retrieved until the class is initialized."
            )

    """
    Operations.
    """

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        parser_help = None
        if len(parser._actions) > 0 and isinstance(
            parser._actions[0], argparse._HelpAction
        ):
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
            "--verbose",  # Note: This is read by SimulationApp through sys.argv
            action="store_true",
            help="Enable verbose-level log output from the SimulationApp.",
        )
        arg_group.add_argument(
            "--info",  # Note: This is read by SimulationApp through sys.argv
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

    def _config_resolution(self, launcher_args: dict):
        """Resolve the input arguments and environment variables.

        Args:
            launcher_args: A dictionary of all input arguments passed to the class object.
        """
        # Handle all control logic resolution

        # --LIVESTREAM logic--
        #
        livestream_env = int(os.environ.get("LIVESTREAM", 0))
        livestream_arg = launcher_args.pop(
            "livestream", AppLauncher._APPLAUNCHER_CFG_INFO["livestream"][1]
        )
        livestream_valid_vals = {0, 1, 2}
        # Value checking on LIVESTREAM
        if livestream_env not in livestream_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `LIVESTREAM`: {livestream_env} ."
                f" Expected: {livestream_valid_vals}."
            )
        # We allow livestream kwarg to supersede LIVESTREAM envvar
        if livestream_arg >= 0:
            if livestream_arg in livestream_valid_vals:
                self._livestream = livestream_arg
                # print info that we overrode the env-var
                print(
                    f"[INFO][AppLauncher]: Input keyword argument `livestream={livestream_arg}` has overridden"
                    f" the environment variable `LIVESTREAM={livestream_env}`."
                )
            else:
                raise ValueError(
                    f"Invalid value for input keyword argument `livestream`: {livestream_arg} ."
                    f" Expected: {livestream_valid_vals}."
                )
        else:
            self._livestream = livestream_env

        # --HEADLESS logic--
        #
        # Resolve headless execution of simulation app
        # HEADLESS is initially passed as an int instead of
        # the bool of headless_arg to avoid messy string processing,
        headless_env = int(os.environ.get("HEADLESS", 0))
        headless_arg = launcher_args.pop(
            "headless", AppLauncher._APPLAUNCHER_CFG_INFO["headless"][1]
        )
        headless_valid_vals = {0, 1}
        # Value checking on HEADLESS
        if headless_env not in headless_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `HEADLESS`: {headless_env} . Expected: {headless_valid_vals}."
            )
        # We allow headless kwarg to supersede HEADLESS envvar if headless_arg does not have the default value
        # Note: Headless is always true when livestreaming
        if headless_arg is True:
            self._headless = headless_arg
        elif self._livestream in {1, 2}:
            # we are always headless on the host machine
            self._headless = True
            # inform who has toggled the headless flag
            if self._livestream == livestream_arg:
                print(
                    f"[INFO][AppLauncher]: Input keyword argument `livestream={self._livestream}` has implicitly"
                    f" overridden the environment variable `HEADLESS={headless_env}` to True."
                )
            elif self._livestream == livestream_env:
                print(
                    f"[INFO][AppLauncher]: Environment variable `LIVESTREAM={self._livestream}` has implicitly"
                    f" overridden the environment variable `HEADLESS={headless_env}` to True."
                )
        else:
            # Headless needs to be a bool to be ingested by SimulationApp
            self._headless = bool(headless_env)
        # Headless needs to be passed to the SimulationApp so we keep it here
        launcher_args["headless"] = self._headless

        # --enable_cameras logic--
        #
        enable_cameras_env = int(os.environ.get("ENABLE_CAMERAS", 0))
        enable_cameras_arg = launcher_args.pop(
            "enable_cameras", AppLauncher._APPLAUNCHER_CFG_INFO["enable_cameras"][1]
        )
        enable_cameras_valid_vals = {0, 1}
        if enable_cameras_env not in enable_cameras_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `ENABLE_CAMERAS`: {enable_cameras_env} ."
                f"Expected: {enable_cameras_valid_vals} ."
            )
        # We allow enable_cameras kwarg to supersede ENABLE_CAMERAS envvar
        if enable_cameras_arg is True:
            self._enable_cameras = enable_cameras_arg
        else:
            self._enable_cameras = bool(enable_cameras_env)
        self._offscreen_render = False
        if self._enable_cameras and self._headless:
            self._offscreen_render = True

        # Check if we can disable the viewport to improve performance
        #   This should only happen if we are running headless and do not require livestreaming or video recording
        #   This is different from offscreen_render because this only affects the default viewport and not other renderproducts in the scene
        self._render_viewport = True
        if (
            self._headless
            and not self._livestream
            and not launcher_args.get("video", False)
        ):
            self._render_viewport = False

        # hide_ui flag
        launcher_args["hide_ui"] = False
        if self._headless and not self._livestream:
            launcher_args["hide_ui"] = True

        # avoid creating new stage at startup by default for performance reasons
        launcher_args["create_new_stage"] = False

        # --simulation GPU device logic --
        self.device_id = 0
        device = launcher_args.get(
            "device", AppLauncher._APPLAUNCHER_CFG_INFO["device"][1]
        )
        if "cuda" not in device and "cpu" not in device:
            raise ValueError(
                f"Invalid value for input keyword argument `device`: {device}."
                " Expected: a string with the format 'cuda', 'cuda:<device_id>', or 'cpu'."
            )
        if "cuda:" in device:
            self.device_id = int(device.split(":")[-1])

        # Raise an error for the deprecated cpu flag
        if launcher_args.get("cpu", False):
            raise ValueError(
                "The `--cpu` flag is deprecated. Please use `--device cpu` instead."
            )

        if "distributed" in launcher_args and launcher_args["distributed"]:
            # local rank (GPU id) in a current multi-gpu mode
            self.local_rank = int(os.getenv("LOCAL_RANK", "0")) + int(
                os.getenv("JAX_LOCAL_RANK", "0")
            )
            # global rank (GPU id) in multi-gpu multi-node mode
            self.global_rank = int(os.getenv("RANK", "0")) + int(
                os.getenv("JAX_RANK", "0")
            )

            self.device_id = self.local_rank
            launcher_args["multi_gpu"] = False
            # limit CPU threads to minimize thread context switching
            # this ensures processes do not take up all available threads and fight for resources
            num_cpu_cores = os.cpu_count()
            num_threads_per_process = num_cpu_cores // int(os.getenv("WORLD_SIZE", 1))
            # set environment variables to limit CPU threads
            os.environ["PXR_WORK_THREAD_LIMIT"] = str(num_threads_per_process)
            os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads_per_process)
            # pass command line variable to kit
            sys.argv.append(
                f"--/plugins/carb.tasking.plugin/threadCount={num_threads_per_process}"
            )

        # set physics and rendering device
        launcher_args["physics_gpu"] = self.device_id
        launcher_args["active_gpu"] = self.device_id

        # Check if input keywords contain an 'experience' file setting
        # Note: since experience is taken as a separate argument by Simulation App, we store it separately
        self._sim_experience_file = launcher_args.pop("experience", "")

        # If nothing is provided resolve the experience file based on the headless flag
        kit_app_exp_path = os.environ["EXP_PATH"]
        isaaclab_app_exp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps"
        )
        if self._sim_experience_file == "":
            # check if the headless flag is setS
            if self._enable_cameras:
                if self._headless and not self._livestream:
                    self._sim_experience_file = os.path.join(
                        isaaclab_app_exp_path, "isaaclab.python.headless.rendering.kit"
                    )
                else:
                    self._sim_experience_file = os.path.join(
                        isaaclab_app_exp_path, "isaaclab.python.rendering.kit"
                    )
            elif self._headless and not self._livestream:
                self._sim_experience_file = os.path.join(
                    isaaclab_app_exp_path, "isaaclab.python.headless.kit"
                )
            else:
                self._sim_experience_file = os.path.join(
                    isaaclab_app_exp_path, "isaaclab.python.kit"
                )
        elif not os.path.isabs(self._sim_experience_file):
            option_1_app_exp_path = os.path.join(
                kit_app_exp_path, self._sim_experience_file
            )
            option_2_app_exp_path = os.path.join(
                isaaclab_app_exp_path, self._sim_experience_file
            )
            if os.path.exists(option_1_app_exp_path):
                self._sim_experience_file = option_1_app_exp_path
            elif os.path.exists(option_2_app_exp_path):
                self._sim_experience_file = option_2_app_exp_path
            else:
                raise FileNotFoundError(
                    f"Invalid value for input keyword argument `experience`: {self._sim_experience_file}."
                    "\n No such file exists in either the Kit or Isaac Lab experience paths. Checked paths:"
                    f"\n\t [1]: {option_1_app_exp_path}"
                    f"\n\t [2]: {option_2_app_exp_path}"
                )
        elif not os.path.exists(self._sim_experience_file):
            raise FileNotFoundError(
                f"Invalid value for input keyword argument `experience`: {self._sim_experience_file}."
                " The file does not exist."
            )

        # Set public IP address of a remote instance
        public_ip_env = os.environ.get("PUBLIC_IP", "127.0.0.1")

        # Process livestream here before launching kit because some of the extensions only work when launched with the kit file
        self._livestream_args = []
        if self._livestream >= 1:
            # Note: Only one livestream extension can be enabled at a time
            if self._livestream == 1:
                logger.warning(
                    "Native Livestream is deprecated. Please use WebRTC Livestream instead with --livestream 2."
                )
                self._livestream_args += [
                    '--/app/livestream/proto="ws"',
                    "--/app/livestream/allowResize=true",
                    "--enable",
                    "omni.kit.livestream.core-4.1.2",
                    "--enable",
                    "omni.kit.livestream.native-5.0.1",
                    "--enable",
                    "omni.kit.streamsdk.plugins-4.1.1",
                ]
            elif self._livestream == 2:
                self._livestream_args += [
                    f"--/app/livestream/publicEndpointAddress={public_ip_env}",
                    "--/app/livestream/port=49100",
                    "--enable",
                    "omni.services.livestream.nvcf",
                ]
            else:
                raise ValueError(
                    f"Invalid value for livestream: {self._livestream}. Expected: 1, 2 ."
                )
            sys.argv += self._livestream_args

        # Resolve additional arguments passed to Kit
        self._kit_args = []
        if "kit_args" in launcher_args:
            self._kit_args = [arg for arg in launcher_args["kit_args"].split()]
            sys.argv += self._kit_args

        # Resolve the absolute path of the experience file
        self._sim_experience_file = os.path.abspath(self._sim_experience_file)
        print(
            f"[INFO][AppLauncher]: Loading experience file: {self._sim_experience_file}"
        )
        # Remove all values from input keyword args which are not meant for SimulationApp
        # Assign all the passed settings to a dictionary for the simulation app
        self._sim_app_config = {
            key: launcher_args[key]
            for key in set(AppLauncher._SIM_APP_CFG_TYPES.keys())
            & set(launcher_args.keys())
        }

    def _create_app(self):
        """Launch SimulationApp"""
        self._app = SimulationApp(
            {
                "headless": self._headless,
                "renderer": self._render_mode,
                "extra_args": ["--/persistent/renderer/rtpt/enabled=true"],
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
