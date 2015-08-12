import numpy as np
import matplotlib.pyplot as plt
%matplotlib
import scipy.ndimage as ndi

import gc_turbulence as g

cp = g.default_cache + 'r13_12_12c.hdf5'

r = g.SingleLayerRun(cache_path=cp)

pp = g.PreProcessor(r)
pp.extract_valid_region()
pp.filter_zeroes()

imp = g.inpainting.Inpainter(pp, scale=pp.front_speed)

# the slice that is causing the hassle
bi = 30
bs = imp.slices[bi]

# the points that the slice is built around
bp = imp.volumes == bi + 1

x = pp.X[bp]
z = pp.Z[bp]
t = pp.T[bp]

# problem is long time axis on invalid slice
# lets try and burst the slice into small slices with a finite time
# size


def burst(s, dt=50):
    sz, sx, st = s
    edges = range(st.start, st.stop, dt) + [st.stop]  # include the end!!
    return [(sz, sx, slice(i, j, None)) for i, j in zip(edges, edges[1:])]
