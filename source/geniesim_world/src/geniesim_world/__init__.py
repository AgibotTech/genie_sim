# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""geniesim_world: ml-sharp add-ons (depth-guided predictor + CLI)."""

from ._version import _resolve_version

from geniesim_world.predictor import DepthGuidedRGBGaussianPredictor, wrap_predictor

__version__ = _resolve_version("geniesim_world")

__all__ = ["DepthGuidedRGBGaussianPredictor", "wrap_predictor", "__version__"]
