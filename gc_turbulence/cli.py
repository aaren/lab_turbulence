"""Command line interface"""
import re
import argparse
import os

from turbulence import SingleLayerFrame
from turbulence import SingleLayerRun


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('command',
                        help="The command that you wish to execute.")
    parser.add_argument('files', nargs='+',
                        help="Pattern to match to exectute command on.")

    args = parser.parse_args()

    commander = Commander(args.files)
    getattr(commander, args.command)()


class Commander(object):
    def __init__(self, items):
        self.items = items

    def list(self):
        for item in self.items:
            print item

    def rename(self):
        """Change folder names from run hash to run index."""
        rename(self.items)

    def import_to_hdf5(self):
        """Import the raw data from each folder to hdf5."""
        for item in self.items:
            pattern = os.path.basename(run_dir)
            run = SingleLayerRun(cache_path=cache_dir,
                                 pattern=pattern,
                                 rex=rex)
            run.import_to_hdf5()


run_index_pattern = \
    re.compile('.*(?P<index>r1[0-9]_[0-9]{2}_[0-9]{2}[a-z]).*')


def rename(directories):
    hash_directories = [d for d in directories
                        if not run_index_pattern.match(d)]

    file_format = 'stereo.{}.000001.csv'.format

    first_files = [os.path.join(hash, file_format(hash)) for hash
                                                        in hash_directories]

    indices = [get_index_from_file(f) for f in first_files]

    for hashd, run_index in zip(hash_directories, indices):
        if run_index:
            index_name = os.path.join(os.path.dirname(hashd), run_index)
            print "rename: {} --> {}".format(hashd, index_name)
            # os.rename(hashd, index_name)
        else:
            pass


def get_index_from_file(filename):
    """Determine the run index from a Dynamic Studio export file."""
    try:
        f = SingleLayerFrame(filename)
    except IOError, TypeError:
        return None

    match = run_index_pattern.match(f.header['Originator'])

    if match:
        index = match.groupdict()['index']
        return index
    else:
        return None
