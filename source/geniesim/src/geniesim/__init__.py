# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""geniesim — agent-driven simulation SDK (meta-package).

This distribution ships **no Python code of its own**. It exists purely as a
PEP 621 meta-package that pulls in the real content-bearing peer
distributions via required dependencies and optional extras:

    geniesim bootstrap                   # required local peers
    pip install -e 'source/geniesim[scene_agent]'  # optional scene tools
    pip install -e 'source/geniesim[all]'          # all optional peers

Modelled on Isaac Sim's layout (``isaacsim`` + ``isaacsim-kernel`` /
``isaacsim-benchmark`` / …): the umbrella name is a stable install target
that delegates the actual code to peers.
"""

from geniesim._version import _resolve_version

__version__: str = _resolve_version("geniesim")

__all__ = ["__version__"]
