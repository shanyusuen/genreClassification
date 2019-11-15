'''
    create_partitions.py

    Creates train/test partitions for classification tasks based on
    label assignments provided by http://www.ifs.tuwien.ac.at/mir/msd/


    Copyright (C) 2012  Alexander Schindler
        Institute of Software Technology and Interactive Systems
        Vienna University of Technology
        Information and Software Engineering Group

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Version 1.0
'''

import os
import gzip
import argparse

if __name__ == '__main__':

    # argument handling
    parser = argparse.ArgumentParser(description='Extract distinct features from an Arff-formatted file')
    parser.add_argument('--outdir', '-o', help='Directory to store output feature files', required=True, type=str)
    parser.add_argument('--featfile', '-f', help='Arff-formatted feature file', required=True, type=str)
    parser.add_argument('--splitfile', '-s', help='Split definition file', required=True, type=str)
    parser.add_argument('--labelfile', '-l', help='File containing track-labels', required=False, type=str)

    # parse arguments
    args = vars(parser.parse_args())

    # assign required variables
    input_file_path = args['featfile']
    label_file_path = args['labelfile']
    split_file_path = args['splitfile']
    destination_path = args['outdir']

    #
    train_file = "{0}/{1}".format(destination_path,
                                  os.path.split(split_file_path.replace(split_file_path.split(".")[-1], "train.arff"))[
                                      -1])
    test_file = "{0}/{1}".format(destination_path,
                                 os.path.split(split_file_path.replace(split_file_path.split(".")[-1], "test.arff"))[
                                     -1])

    # open output files
    train = open(train_file, "w")
    test = open(test_file, "w")

    # load split mapping
    split_file = open(split_file_path, 'r')
    split_mapping = {}

    for line in split_file:

        # skip comments
        if line[0] == "%":
            train.write("{0}\n".format(line.rstrip()))
            test.write("{0}\n".format(line.rstrip()))
            continue

        tmp = line.rstrip().split("\t")
        try:
            split_mapping[tmp[0]] = tmp[1]
        except:
            print("failed to add: ", tmp)
            print("continuing")


    split_file.close()

    # load class label assignments
    label_file = open(label_file_path, 'r')
    labels_mapping = {}
    labels = set()
    skip_line = True

    # load labels
    for line in label_file:

        # skip header line
        if skip_line:
            skip_line = False
            continue

        tmp = line.rstrip().split("\t")
        labels_mapping[tmp[0]] = tmp[1]
        labels.add(tmp[1])

    label_file.close()

    # add newline
    train.write("\n")
    test.write("\n")

    # process feature file
    if input_file_path.split(".")[-1] == "gz":
        track_id_file = gzip.open(input_file_path, 'r')
    else:
        track_id_file = open(input_file_path, 'r')

    header = True

    for line in track_id_file:

        # copy header
        if header:

            if line.lower().rstrip() == "@data":
                header = False

            if line.find("string") != -1:

                labels_str = ""

                for label in labels:
                    labels_str = "{0},{1}".format(labels_str, label)

                labels_str = labels_str[1:]

                train.write("@ATTRIBUTE class {{{0}}}\n".format(labels_str))
                test.write("@ATTRIBUTE class {{{0}}}\n".format(labels_str))

            else:

                train.write("{0}\n".format(line.rstrip()))
                test.write("{0}\n".format(line.rstrip()))

            continue

        # process feature vector
        track_id = line.lstrip().split(",")[-1].strip()

        if len(track_id) < 5:
            track_id = line.lstrip().split(",")[-2].strip()

        if track_id[0].lstrip() == "'":
            track_id = track_id.lstrip()[1:-1]

        if track_id in split_mapping and track_id in labels_mapping:

            if split_mapping[track_id].lower() == "train":
                train.write("{0}\n".format(line.rstrip().replace(track_id, labels_mapping[track_id])))

            elif split_mapping[track_id].lower() == "test":
                test.write("{0}\n".format(line.rstrip().replace(track_id, labels_mapping[track_id])))

            else:
                print()
                "*** ERROR: Unexpected label in track assignment file"

    train.close()
    test.close()

    track_id_file.close()
