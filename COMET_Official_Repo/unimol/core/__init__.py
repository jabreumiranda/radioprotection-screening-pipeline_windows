# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from unimol.core.X import Y`
from unimol.core.distributed import utils as distributed_utils
from unimol.core.logging import meters, metrics, progress_bar  # noqa

sys.modules["unimol.core.distributed_utils"] = distributed_utils
sys.modules["unimol.core.meters"] = meters
sys.modules["unimol.core.metrics"] = metrics
sys.modules["unimol.core.progress_bar"] = progress_bar

import unimol.core.losses  # noqa
import unimol.core.distributed  # noqa
import unimol.core.models  # noqa
import unimol.core.modules  # noqa
import unimol.core.optim  # noqa
import unimol.core.optim.lr_scheduler  # noqa
import unimol.core.tasks  # noqa

