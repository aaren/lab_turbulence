import time

from gc_turbulence.util import parallel_process, parallel_stub

@parallel_stub
def f(kwargs):
    pbar = kwargs['pbar']
    i = kwargs['i']
    pbar.update()
    time.sleep(1)
    return i**2

parallel_process(f, [{'i': i} for i in range(10)], N=10)
