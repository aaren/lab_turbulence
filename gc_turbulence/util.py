import functools
import os
import errno
import multiprocessing as mp
from multiprocessing.managers import BaseManager

from progressbar import ProgressBar


def makedirs_p(path):
    """Emulate mkdir -p. Doesn't throw error if directory
    already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class ProgressUpdater(object):
    """Wrapper around the ProgressBar class to use as a shared
    progress bar.

    Notably, has an update() method that does not require any
    argument so that it can be called by functions that don't
    know where they are in the order of execution. Useful for
    multiprocessing.
    """
    def __init__(self, maxval=None):
        self.pbar = ProgressBar(maxval=maxval)

    def start(self):
        self.pbar.start()

    def finish(self):
        self.pbar.finish()

    def update(self, i=None):
        if not i:
            i = self.pbar.currval + 1
        self.pbar.update(i)

    def currval(self):
        return self.pbar.currval


class ProgressManager(BaseManager):
    """Create custom manager"""
    pass


# register the progressbar with the new manager
ProgressManager.register(typeid='ProgressUpdater',
                         callable=ProgressUpdater,
                         exposed=['start', 'finish', 'update', 'currval'])


def parallel_process(function, kwarglist, N, processors=None):
    """Parallelise execution of a function over a list of arguments.

    Inputs: function - function to apply
            kwarglist - iterator of keyword arguments to apply
                        function to
            processors - number of processors to use, default None
                         is to use 4 times the number of processors

    Returns: an *unordered* list of the return values of the
             function.

    The list of arguments must be formatted as keyword arguments,
    i.e. be a list of dictionaries.
    TODO: can it actually be a generator?
    No because take the len() of it.

    Explicitly splits a list of inputs into chunks and then
    operates on these chunks one at a time.

    I have tried using Pool for this, but it doesn't seem to release
    memory sensibly, despite setting maxtasksperchild. Thus, I've
    used Process and explicitly start and end the jobs.

    This does what it was supposed to which is keep the memory usage
    limited to that needed by the number of calls that can fit into
    the number of processes.  However, it is pretty slow if you have
    processors equal to the number available. If you set it high things
    happen quicker, but the load average can go a bit mad :).
    """
    if not processors:
        processors = 2

    # shared progressbar
    progress_manager = ProgressManager()
    progress_manager.start()
    pbar = progress_manager.ProgressUpdater(maxval=N)
    pbar.start()

    # manager = mp.Manager()
    kwargs_list = [dict(a, pbar=pbar) for a in kwarglist]

    pool = mp.Pool()
    outputs = pool.imap_unordered(function, kwargs_list)
    pool.close()
    pool.join()
    pbar.finish()

    return outputs


def parallel_stub(stub):
    """Decorator to use on functions that are fed to
    parallel_process. Calls the function and appends any output to
    the queue, then updates the progressbar.
    """
    @functools.wraps(stub)
    def f(kwargs):
        pbar = kwargs.pop('pbar')
        ret = stub(**kwargs)
        pbar.update()
        return ret
    return f
