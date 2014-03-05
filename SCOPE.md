Aim: Store more attributes in the data import.

Currently:

Each run occupies a single hdf5 file. The file contains a dump of
all of the data from the run. The only identifying feature of the
file is the file name, which corresponds to a unique hash that is
associated with the run.

TODO:

- Extract run info from parameters file
- Add each of the parameters as an attribute to the hdf5
- name the hdf5 using the run index, rather than the hash


We need to know which run index each hash is associated with. Each
data file has the index as a string in its header info (this was a
convention when making the measurements).

The index is a regular expression of form

    '.*(r1[0-9]_[0-9][0-9]_[0-9][0-9][a-z]).*

i.e. something like 'r13_01_15e'.


### Raw data

Dynamic Studio outputs each run as a series of csv files. Each file contains
velocity data from a single pair of PIV images.

Each series of files follows the naming convention

    'stereo.${hash}.${number}.csv'

where ${hash} is the unique run hash and ${number} is the number of
the velocity data frame. The run hash is not found anywhere inside the files
and is found only in the filenames. We would like to preserve this as well.

We can verify that each series of files starts with the number '000001'
by comparing the output of `find` in the directory containing the raw data.
The number of runs is:
    
    find . -type d | wc -l

And the number of '000001' files is:

    find . -type f -name *000001.csv


### Renaming the folders

First step: rename the folders containing the raw data from ${hash} 
to ${run_index}.

```python
import re
import os

import gc_turbulence as g

raw_data = '/home/eeaol/lab/data/flume2/main_data/raw'

run_index_pattern = re.compile('.*(?P<index>r1[0-9]_[0-9]{2}_[0-9]{2}[a-z]).*')

directories = os.listdir(raw_data)

hash_directories = [d for d in directories if not run_index_pattern.match(d)]

file_format = 'stereo.{}.000001.csv'.format

first_file_list = [os.path.join(raw_data, hash, file_format(hash)) for hash in hash_directories]

def get_index_from_file(filename):
    f = g.SingleLayerFrame(filename)
    match = run_index_pattern.match(f.header['Originator'])
    index = match.groupdict['index']
    return index
```


### Command line interface

Let's make all of this usable through a command line interface to
the gc_turbulence library.

If we want to rename folders in the raw directory to their run index
we do

    tt rename pattern

where pattern is a regex to match folder names.

If we want to import runs in a directory we do

    tt import pattern

where pattern is a regex to match folder names.
