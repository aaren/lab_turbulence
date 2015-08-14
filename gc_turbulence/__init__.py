from exceptions import DeprecationWarning
import warnings

from . import attributes
from .import runbase
from .import processing
from .import analysis
from .import util

from .runbase import RawFrame, RawRun

from .processing import ProcessedRun
from .processing import PreProcessor

from .analysis import AnalysisRun

from .attributes import Parameters

from .analysis import DMD

from .commander import cli

from .config import *


def SingleLayerFrame(*args, **kwargs):
    warnings.warn('SingleLayerFrame is now called RawRun', DeprecationWarning)
    return RawFrame(*args, **kwargs)


def SingleLayerRun(*args, **kwargs):
    warnings.warn('SingleLayerRun is now called RawRun', DeprecationWarning)
    return RawRun(*args, **kwargs)
