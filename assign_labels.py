'''
    assign_labels.py

    Assign genre lables based on label assignments provided by
    http://www.ifs.tuwien.ac.at/mir/msd/ and remove unlabelled
    instances.

    Copyright (C) 2013  Alexander Schindler
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
    parser = argparse.ArgumentParser(description='Assign gorund truth labels to feature file',
                                     epilog="Only labelled instances will be included in output file")
    parser.add_argument('--featfile', '-f', help='Arff-formatted feature file', required=True, type=str)
    parser.add_argument('--labelfile', '-l', help='File containing track-labels', required=False, type=str)
    parser.add_argument('--outdir', '-o', help='Directory to store output feature files', required=True, type=str)
    parser.add_argument('--compress', '-c', help='gzip the output file', action='store_true')

    # parse arguments
    args = vars(parser.parse_args())

    # assign required variables
    input_file_path = args['featfile'].replace("\\", "/")
    label_file_path = args['labelfile'].replace("\\", "/")
    destination_path = args['outdir']
    compress = args["compress"]

    #
    out_file = "{0}/{1}".format(destination_path, "{0}_{1}.arff".format(input_file_path.split("/")[-1].split(".")[0],
                                                                        label_file_path.split("/")[-1].split(".")[0]))

    # open output files
    if compress:
        output = gzip.open("{0}.gz".format(out_file), "w")
    else:
        output = open(out_file, "w")

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

                # if line.find("string") != -1:

                labels_str = ""

                for label in labels:
                    labels_str = "{0},{1}".format(labels_str, label)

                labels_str = labels_str[1:]

                output.write("@ATTRIBUTE class {{{0}}}\n".format(labels_str))
                output.write("\n@data\n")

            else:

                if line.find("instanceName") == -1:
                    output.write("{0}\n".format(line.rstrip()))

            continue

        # process feature vector
        track_id = line.lstrip().split(",")[-1].strip()

        if len(track_id) < 5:
            track_id = line.lstrip().split(",")[-2].strip()

        if track_id[0].lstrip() == "'":
            track_id = track_id.lstrip()[1:-1]

        if labels_mapping.has_key(track_id):
            output.write("{0}\n".format(line.rstrip().replace(track_id, labels_mapping[track_id]).strip()[:-1]))

    output.close()

    track_id_file.close()
