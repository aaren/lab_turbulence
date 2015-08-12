import os

default_root = os.environ['HOME'] + '/lab/data/flume2/main_data/'
default_cache = default_root + 'cache/'
default_processed = default_root + 'processed/'

single_layer_parameters = os.path.join(default_root, 'params_single_layer')
two_layer_parameters = os.path.join(default_root, 'params_two_layer')
