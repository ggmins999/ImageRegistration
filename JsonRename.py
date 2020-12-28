#!/usr/bin/env python3
import argparse
import shutil

# Create the parser
parser = argparse.ArgumentParser(prog='renamefiles', description='renames json files and image files')

parser.add_argument('--prefix', help = 'prefix to name file')
parser.add_argument('--inputdir', help='directory from which you want to take')
parser.add_argument('--outputdir', help='new renamed directory')


args = parser.parse_args()

import os

path = args.inputdir
files = os.listdir(path)
prefix = args.prefix
print(f'path={path}, files={files}, prefix={prefix}')
outpath = args.outputdir

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

for index, file in enumerate(files):
    if os.path.isfile(os.path.join(path, file)):
        if file.lower().endswith('png') or file.lower().endswith('jpg') or file.lower().endswith('jpeg'):
            shutil.copy(os.path.join(path, file), os.path.join(outpath, ''.join([prefix, "_", file])))

import json
# # this finds our json files

json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]

# # we need both the json and an index number so use enumerate()
for filename in (json_files):
    if not os.path.isfile(os.path.join(path, filename)):
        continue
    with open(os.path.join(path, filename)) as json_file:
        data = json.load(json_file)
        data['imagePath'] = prefix + "_" + data['imagePath']
    with open(os.path.join(outpath, prefix + "_" + filename), 'w') as new_json_file:
        json.dump(data, new_json_file)

