"""Command line interface"""
import re
import argparse
import os
import glob
import logging

import h5py

from turbulence import SingleLayerFrame
from turbulence import SingleLayerRun
from turbulence import PreProcessor
from turbulence import Parameters


logging.basicConfig(filename='process.log', level=logging.DEBUG)


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
        """Change folder names from run hash to run index.

        Looks at each folder and determines whether the name is
        a hash or an index. If a hash, finds the index by looking
        at the first file in the folder and renames the folder to
        the index.
        """
        renamer = Renamer()
        hash_directories = [d for d in self.items if not
                                                    renamer.is_pattern_dir(d)]
        for each in hash_directories:
            renamer.rename(each)

    def assimilate(self, args=''):
        """Import the raw data from each folder to hdf5."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--rex', default=None,
                            help="regular expression to match files")
        parser.add_argument('--pattern', default='*',
                            help="regular expression to match files")
        parser.add_argument('--cache', default='cache',
                            help="directory to save the cache files to")
        parser.add_argument('--new', action='store_true',
                            help="Only import the unimported.")
        args = parser.parse_args(args)

        parameters = Parameters()

        cached = [os.path.splitext(c)[0] for c in os.listdir(args.cache)]

        for item in self.items:
            index = os.path.basename(item)
            if index in cached and args.new:
                pass
            else:
                try:
                    self.assim(item, args, parameters)
                except:
                    print("Failed to extract {}".format(item))

    @staticmethod
    def assim(item, args, parameters):
        run = SingleLayerRun(data_dir=item,
                             pattern=args.pattern,
                             rex=args.rex)
        run.init_load_frames()

        folder_name = os.path.basename(item)
        run_info = parameters(folder_name)
        if not run_info:
            print "\n!!! Could not find info for %s !!!\n" % folder_name
            run_info = {}

        # the run hash is found embedded in the file path
        hash_pattern = re.compile(r'.*stereo\.(?P<hash>.*)\.(?P<num>.*)\.csv')
        # extract and store the run hash
        rhash = hash_pattern.match(run.files[0]).groupdict()['hash']
        run_info['hash'] = rhash

        run.cache_path = '{}/{}.hdf5'.format(args.cache, folder_name)
        info = "Importing to hdf5: {} --> {} ({} files)"
        print info.format(item, run.cache_path, run.nfiles)
        run.import_to_hdf5(attributes=run_info)

    def info(self):
        """Find information about things."""
        renamer = Renamer()
        for item in self.items:
            # try and get the attributes if hdf5
            if h5py.is_hdf5(item):
                self.hdf5_info(item)

            # then try and load an index from the paramters file
            elif renamer.pattern_match(item):
                self.run_info(item)

            # if it is a file, try to make a frame out of it
            elif os.path.isfile(item):
                self.file_info(item)
            # then try and find an index in the first file of the
            # item
            else:
                fpattern = renamer.first_file_format('*')
                fname = os.path.join(item, fpattern)
                path = glob.glob(fname)[0]
                self.file_info(path)

    def file_info(self, item):
        renamer = Renamer()
        index = renamer.get_index_from_file(item)
        if index:
            self.run_info(index)
        else:
            self.frame_info(item)

    def frame_info(self, item):
        f = SingleLayerFrame(item)
        for k, v in f.header.items():
            print "{}: {}".format(k, v)

    def hdf5_info(self, item):
        """Display information about the contents of a hdf5 file."""
        if not h5py.is_hdf5(item):
            print "{} is not hdf5".format(item)
            exit()

        print "{}: ".format(item)
        info_line = "\t{}".format
        h5 = h5py.File(item, 'r')
        print "\t### Datasets:"
        for k in h5.keys():
            print info_line(h5[k])

        print "\n\t### Attributes:"
        for k, v in h5.attrs.items():
            print info_line("{}: {}".format(k, v))

    def run_info(self, item):
        """Display the run info for a given run index.
        If given a list of files, this will use the basename
        as the index.

        If the given string does not match the run index pattern
        then nothing is returned.
        """
        params = Parameters()
        renamer = Renamer()

        index = renamer.pattern_match(item).groupdict()['index']
        run_info = params(index)
        info_line = "{}: {}".format
        if run_info:
            for k, v in run_info.items():
                print info_line(k, v)

    def pre_process(self, args=''):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', default='processed',
                            help="directory to save the pre-processed data to")
        args = parser.parse_args(args)

        for item in self.items:
            if not h5py.is_hdf5(item):
                print "{} is not hdf5!".format(item)

            else:
                try:
                    self._pre_process(item, args.output)
                except Exception:
                    print("Failed to process {}".format(item))
                    logging.exception('Could not process {}'.format(item))

    @staticmethod
    def _pre_process(item, outdir):
        fname = os.path.basename(item)
        outpath = os.path.join(outdir, fname)

        run = SingleLayerRun(cache_path=item)
        pp = PreProcessor(run=run)
        print "Pre-processing {} ...".format(item)
        pp.execute()
        print "writing data to {} ...".format(outpath)
        pp.write_data(outpath)


class Renamer(object):
    """Collection of tools for renaming runs."""
    run_index_pattern = \
        re.compile(r'.*(?P<index>r1[0-9]_[0-9]{2}_[0-9]{2}[a-z]).*')

    first_file_format = 'stereo.{}.000001.csv'.format

    def pattern_match(self, string):
        return self.run_index_pattern.match(string)

    def is_pattern_dir(self, directory):
        """Determine whether a directory / filename (string) matches
        the regex for a run index."""
        if self.pattern_match(directory):
            return True
        else:
            return False

    def rename(self, directory):
        """Rename a directory that has the run hash as the basename"""
        hash = os.path.basename(directory)
        first_file = os.path.join(directory, self.first_file_format(hash))
        run_index = self.get_index_from_file(first_file)
        if run_index:
            index_name = os.path.join(os.path.dirname(directory), run_index)
            print "rename: {} --> {}".format(directory, index_name)
            os.rename(directory, index_name)
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
