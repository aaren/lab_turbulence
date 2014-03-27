import os

import turbulence
import analysis
import util

from turbulence import SingleLayerFrame, SingleLayerRun
from turbulence import ProcessedRun
from turbulence import PreProcessor
from turbulence import Parameters
from analysis import DMD
from analysis import Histograms

from commander import cli

default_root = os.environ['HOME'] + '/lab/data/flume2/main_data/'
default_cache = default_root + 'cache/'
default_processed = default_root + 'processed/'
