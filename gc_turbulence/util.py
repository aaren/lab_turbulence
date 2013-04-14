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


def parallel_process(function, arglist, processors=None):
    """Parallelise execution of a function over a list of arguments.

    Inputs: function - function to apply
            arglist - iterator of arguments to apply function to
            processors - number of processors to use, default None is
                         to use 4 times the number of processors

    Explicitly splits a list of inputs into chunks and then
    operates on these chunks one at a time.

    I have tried using Pool for this, but it doesn't seem to release
    memory sensibly, despite setting maxtasksperchild.

    This function does not return an ordered list of outputs.

    This does what it was supposed to which is keep the memory usage
    limited to that needed by the number of calls that can fit into
    the number of processes.  However, it is pretty slow if you have
    processors equal to the number available. If you set it high things
    happen quicker, but the load average can go a bit mad :).


    """
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

    # set processors equal to 4 times the number of physical
    # processors - typically good trade between memory use and
    # waiting for processes to get going
    if not processors:
        processors = mp.cpu_count() * 4

    # shared progressbar
    progress_manager = ProgressManager()
    progress_manager.start()
    N = len(arglist)
    pbar = progress_manager.ProgressUpdater(maxval=N)
    pbar.start()

    # The queue for storing the results
    # manager = mp.Manager()
    queue = mp.Queue()
    args = [dict(a, queue=queue, pbar=pbar) for a in arglist]

    outputs = []
    for job in chunker(args, processors):
        processes = [mp.Process(target=function, args=(arg,)) for arg in job]
        # start them all going
        for p in processes:
            p.start()
        # populate the queue, if it was given
        if queue:
            for p in processes:
                outputs.append(queue.get())
        else:
            outputs = None
        # now wait for all the processes in this job to finish
        for p in processes:
            p.join()

    pbar.finish()
    return outputs
