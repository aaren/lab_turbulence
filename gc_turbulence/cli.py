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
    parser.add_argument('files', nargs='*',
                        help="Pattern to match to exectute command on.")

    args, remainder = parser.parse_known_args()
    commander = Commander(args.files)

    if len(remainder) > 0:
        getattr(commander, args.command)(remainder)

    else:
        getattr(commander, args.command)()


class Commander(object):
    def __init__(self, items):
        self.items = items

    def list(self):
        for item in self.items:
            print item

    def rename(self):
        """Change folder names from run hash to run index."""
        renamer = Renamer()
        hash_directories = [d for d in self.items if not
                                                    renamer.is_pattern_dir(d)]
        for each in hash_directories:
            renamer.rename(each)

    def import_to_hdf5(self, args=''):
        """Import the raw data from each folder to hdf5."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--rex', default=None,
                            help="regular expression to match files")
        parser.add_argument('--pattern', default='*',
                            help="regular expression to match files")
        args = parser.parse_args(args)

        for item in self.items:
            cache_path = 'cache/{}.hdf5'.format(args.pattern)
            run = SingleLayerRun(data_dir=item,
                                 cache_path=cache_path,
                                 pattern=args.pattern,
                                 rex=args.rex)
            run.init_load_frames()
            print run.nfiles
            # run.import_to_hdf5()


class Renamer(object):
    """Collection of tools for renaming runs."""
    run_index_pattern = \
        re.compile('.*(?P<index>r1[0-9]_[0-9]{2}_[0-9]{2}[a-z]).*')

    file_format = 'stereo.{}.000001.csv'.format

    def is_pattern_dir(self, directory):
        if self.run_index_pattern.match(directory):
            return True
        else:
            return False

    def rename(self, directory):
        """Rename """
        hash = os.path.basename(directory)
        first_file = os.path.join(hash, self.file_format(hash))
        run_index = self.get_index_from_file(first_file)
        if run_index:
            index_name = os.path.join(os.path.dirname(directory), run_index)
            print "rename: {} --> {}".format(hash, index_name)
            # os.rename(hashd, index_name)
        else:
            pass

    def get_index_from_file(self, filename):
        """Determine the run index from a Dynamic Studio export file."""
        try:
            f = SingleLayerFrame(filename)
        except (IOError, TypeError):
            return None

        match = self.run_index_pattern.match(f.header['Originator'])

        if match:
            index = match.groupdict()['index']
            return index
        else:
            return None
