import os
import errno

from parallelprogress import parallel_process, parallel_stub


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
