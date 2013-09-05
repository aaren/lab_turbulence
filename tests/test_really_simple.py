import time

import multiprocessing as mp

from gc_turbulence.util import parallel_process, parallel_stub

@parallel_stub
def my_f(i):
    time.sleep(1)
    return i**2

kwargs = [{'i': i} for i in range(10)]

# output = parallel_process(my_f, kwargs, N=10)
# print output

def my_other_f(i):
    time.sleep(1)
    return i**2

pool = mp.Pool()

output = pool.map(my_other_f, [e['i'] for e in kwargs])
print output
