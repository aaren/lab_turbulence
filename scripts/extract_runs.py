import sys
import glob
import os

from gc_turbulence import SingleLayerRun


working_dir = '/home/eeaol/lab/data/flume2/main_data'
cache_dir = '/home/eeaol/lab/data/flume2/main_data/cache'


def extract(run_dir, rex):
    pattern = os.path.basename(run_dir)
    run = SingleLayerRun(data_dir=run_dir,
                         cache_path=cache_dir,
                         pattern=pattern,
                         rex=rex)
    run.import_to_hdf5()

pattern = sys.argv[1]
if len(sys.argv) > 2:
    rex = sys.argv[2]
else:
    rex = '*'

run_dirs = glob.glob(os.path.join(working_dir, pattern))
print "Found {n} runs: \n".format(n=len(run_dirs))

for run in run_dirs:
    print run

print "\nExtract? (y/n)"

query = raw_input('> ')
if query == 'y':
    for run_dir in run_dirs:
        print "Extracting {}...".format(run_dir)
        extract(run_dir, rex)
else:
    exit('Bye')
