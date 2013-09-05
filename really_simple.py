import time

from gc_turbulence.util import parallel_process, parallel_stub

@parallel_stub
def my_f(i):
    time.sleep(1)
    return i**2

parallel_process(my_f, [{'i': i} for i in range(10)], N=10)
