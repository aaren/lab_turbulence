from collections import OrderedDict

import numpy as np

import config


class Parameters(object):
    """For a given run index, determine the type of run
    (single layer / two layer) and load the appropriate
    parameters.

    An instance of this class is a function that returns
    a (ordered) dictionary of the run parameters.
    """
    # TODO: this path should be in init or config or somewhere else
    root = config.default_root
    single_layer_parameters = config.single_layer_parameters
    two_layer_parameters = config.two_layer_parameters

    single_layer_headers = [('run_index',        '|S10'),
                            ('H',                np.float),
                            ('D',                np.float),
                            ('L',                np.float),
                            ('rho_ambient',      np.float),
                            ('rho_lock',         np.float),
                            ('T_ambient',        np.float),
                            ('T_lock',           np.float),
                            ('n_sample_ambient', np.float),
                            ('n_sample_lock',    np.float),
                            ('T_sample_ambient', np.float),
                            ('T_sample_lock',    np.float),
                            ]

    two_layer_headers = [('run_index',      '|S10'),
                         ('H',              np.float),
                         ('D',              np.float),
                         ('L',              np.float),
                         ('h_1 / H',        np.float),
                         ('rho_upper',      np.float),
                         ('rho_lower',      np.float),
                         ('rho_lock',       np.float),
                         ('T_upper',        np.float),
                         ('T_lower',        np.float),
                         ('T_lock',         np.float),
                         ('n_sample_upper', np.float),
                         ('n_sample_lower', np.float),
                         ('n_sample_lock',  np.float),
                         ('T_sample_upper', np.float),
                         ('T_sample_lower', np.float),
                         ('T_sample_lock',  np.float),
                         ]

    def __init__(self, run_index=None):
        self.init_parameters()

    def __call__(self, run_index):
        run_type = self.determine_run_type(run_index)
        if run_type == 'single layer':
            return self.get_run_info(self.single_layer, run_index)
        elif run_type == 'two layer':
            return self.get_run_info(self.two_layer, run_index)
        else:
            return None

    def get_run_info(self, parameters, index):
        """Return the info for a given run index as an
        OrderedDict.
        """
        line = np.where(parameters['run_index'] == index)
        info = parameters[line]
        keys = info.dtype.names
        values = info[0]
        odict = OrderedDict(zip(keys, values))
        odict['run type'] = self.determine_run_type(index)
        return odict

    def determine_run_type(self, run_index):
        """Returns the run type as a string, either
        'two_layer' or 'single_layer'.
        """
        if run_index in self.single_layer['run_index']:
            return 'single layer'
        elif run_index in self.two_layer['run_index']:
            return 'two layer'
        else:
            return None

    def init_parameters(self):
        """Load the parameters files."""
        self.single_layer = self.load_parameters(self.single_layer_parameters,
                                                 self.single_layer_headers)
        self.two_layer = self.load_parameters(self.two_layer_parameters,
                                              self.two_layer_headers)

    @staticmethod
    def load_parameters(file, headers):
        return np.loadtxt(file, dtype=headers, skiprows=2)


class BaseAttributes(object):
    # data vertical step (m)
    dz = 0.00116
    # data horizontal step (m)
    dx = 0.00144
    # data time step (s)
    dt = 0.01


class ProcessedVectors(object):
    # the names of the attributes that an instance should have
    # after running self.execute()
    vectors = [('X',  np.float32),  # streamwise coordinates
               ('Z',  np.float32),  # vertical coordinates
               ('T',  np.float32),  # time coordinates
               ('U',  np.float32),  # streamwise velocity
               ('V',  np.float32),  # cross stream velocity
               ('W',  np.float32),  # vertical velocity
               ]
    vectors = np.dtype(vectors)


class AnalysisVectors(object):
    vectors = [('X',  np.float32),  # streamwise coordinates
               ('Z',  np.float32),  # vertical coordinates
               ('T',  np.float32),  # time coordinates

               ('x',  np.float32),  # streamwise coordinates (single vector)
               ('z',  np.float32),  # vertical coordinates (single vector)
               ('t',  np.float32),  # time coordinates (single vector)

               ('U',  np.float32),  # streamwise velocity
               ('V',  np.float32),  # cross stream velocity
               ('W',  np.float32),  # vertical velocity

               ('fx', np.float32),  # front detection in space
               ('ft', np.float32),  # front detection in time
               ('front_speed', np.float32),     # LAB coord front speed

               # Waves (LAB coords)
               # (U = U - wU - wUr - Ubg)
               ('wU',  np.float32),  # streamwise waves (fitted)
               ('wW',  np.float32),  # vertical waves (fitted)

               ('wUr',  np.float32),  # streamwise waves (remainder)
               ('wWr',  np.float32),  # vertical waves (remainder)

               ('Ubg',  np.float32),  # background in u
               ('Wbg',  np.float32),  # background in w

               # __FRONT coords
               ('Xf', np.float32),  # front relative streamwise coords
               ('Zf', np.float32),  # front relative vertical coords
               ('Tf', np.float32),  # front relative time coords

               ('xf',  np.float32),  # streamwise coordinates (single vector)
               ('zf',  np.float32),  # vertical coordinates (single vector)
               ('tf',  np.float32),  # time coordinates (single vector)

               ('Uf', np.float32),  # front relative streamwise velocity
               ('Wf', np.float32),  # front relative vertical velocity

               ('t0', np.float32),  # front detection in time (scalar)
               ]
    vectors = np.dtype(vectors)


class ProcessorAttributes(BaseAttributes):
    """Class attributes that define vector names / types and
    measurements used in data processing.

    Intended to be inherited by all classes that require access to
    these attributes.
    """
    # The origin of the coordinate system (centre of the
    # calibration target) is 105mm from the base of the tank
    # and 3250mm from the back of the lock.
    # Coordinates are x positive downstream (away from the lock)
    # and z positive upwards (away from the base).
    horizontal_offset = 3.250
    vertical_offset = 0.105

    # In the calibration coordinate system, the valid region
    # is a rectangle with lower left (-0.06, -0.10) and upper
    # right (0.10, 0.02).
    # TODO: do lock relative transform first and change these to
    # lock relative coords
    valid_region_xlim = (-0.070, 0.09)
    valid_region_ylim = (-0.094, 0.02)

    save_vectors = ProcessedVectors.vectors


class ProcessedAttributes(BaseAttributes):
    vectors = ProcessedVectors.vectors
    save_vectors = AnalysisVectors.vectors


class AnalysisAttributes(BaseAttributes):
    vectors = AnalysisVectors.vectors
