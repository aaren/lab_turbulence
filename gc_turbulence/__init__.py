from exceptions import DeprecationWarning
import warnings

from . import attributes
from . import runbase

from . import processing

from . import analysis
from . import plots

from . import util

from .runbase import RawFrame, RawRun

from .processing import ProcessedRun
from .processing import PreProcessor

from .analysis import AnalysisRun

from .attributes import Parameters

from .commander import cli

from .config import (default_root,
                     default_cache,
                     default_processed,
                     default_analysis)

from config import single_layer_parameters, two_layer_parameters


default_paths = {'raw': default_cache,
                 'processed': default_processed,
                 'analysis': default_analysis}

run_types = {'raw': RawRun,
             'processed': ProcessedRun,
             'analysis': AnalysisRun}


def SingleLayerFrame(*args, **kwargs):
    warnings.warn('SingleLayerFrame is now called RawRun', DeprecationWarning)
    return RawFrame(*args, **kwargs)


def SingleLayerRun(*args, **kwargs):
    warnings.warn('SingleLayerRun is now called RawRun', DeprecationWarning)
    return RawRun(*args, **kwargs)


def load(index, kind='analysis', load=True):
    """Create and return a Run instance for a specific index."""
    path = default_paths[kind]
    Run = run_types[kind]

    cache_path = path + index + '.hdf5'

    return Run(cache_path=cache_path, load=load)
