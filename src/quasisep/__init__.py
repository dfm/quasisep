# -*- coding: utf-8 -*-

__all__ = [
    "__version__",
    "DiagQSM",
    "StrictLowerTriQSM",
    "StrictUpperTriQSM",
    "LowerTriQSM",
    "UpperTriQSM",
    "SquareQSM",
    "SymmQSM",
]

from quasisep.quasisep import (
    DiagQSM,
    LowerTriQSM,
    SquareQSM,
    StrictLowerTriQSM,
    StrictUpperTriQSM,
    SymmQSM,
    UpperTriQSM,
)
from quasisep.quasisep_version import version as __version__

__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__uri__ = "https://github.com/dfm/quasisep"
__license__ = "Apache 2"
__description__ = "Quasiseparable matrices in JAX"
__copyright__ = "2022 Simons Foundation, Inc."
__contributors__ = "https://github.com/dfm/quasisep/graphs/contributors"
__bibtex__ = __citation__ = """TBD"""
