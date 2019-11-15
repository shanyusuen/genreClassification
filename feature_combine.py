'''
    feature_combine.py


'''

import os
import gzip
import argparse
#requires liac-arff
import arff
import numpy as np

if __name__ == '__main__':

    # argument handling
    parser = argparse.ArgumentParser(description='Assign gorund truth labels to feature file',
                                     epilog="Only labelled instances will be included in output file")
    parser.add_argument('--featfile1', '-f1', help='Arff-formatted feature file', required=True, type=str)
    parser.add_argument('--featfile2', '-f2', help='Arff-formatted feature file', required=True, type=str)
    parser.add_argument('--outdir', '-o', help='Directory to store output feature files', required=True, type=str)

    # parse arguments
    args = vars(parser.parse_args())

    # assign required variables
    input_file_path1 = args['featfile1'].replace("\\", "/")
    input_file_path2 = args['featfile2'].replace("\\", "/")
    destination_path = args['outdir']


    data1 = arff.load(open(input_file_path1, 'rb'))
    data2 = arff.load(open(input_file_path2, 'rb'))

    new_data = dict()
    new_data['relation'] = '_'.join([data1['relation'], data2['relation']])
    new_data['description'] = " ".join(["This is a combination of the features of", input_file_path1, input_file_path2])

    new_data['attributes'] = []


    ## Concatenate Attributes

    for attr in data1['attributes']:
        #TODO: return this to correct tag
        #if attr[0] != 'MSD_TRACKID':
        attr = list(attr)
        if attr[0] != 'tag':
            attr[0] = ''.join((attr[0],"_1"))
            new_data['attributes'].append(attr)

    for attr in data2['attributes']:
        attr = list(attr)
        #if attr[0] != 'MSD_TRACKID':
        if attr[0] != 'tag':
            attr[0] = ''.join((attr[0], "_2"))
            new_data['attributes'].append(attr)
        else:
            new_data['attributes'].append(attr)


    ## Concatenate Data

    #2d numpy array of data. NXD
    data_array1 = np.array(data1['data'])
    data_array2 = np.array(data2['data'])

    new_entries = []
    #d1 is a single row entry
    for d1 in data_array1:
        d2 = []
        track_label = d1[-1]
        for row in data_array2:
            if row[-1] == track_label:
                d2 = row
                break

        #if the track is matched in both files, append the new data with all features
        if len(d2) != 0:
            new_entry = []
            for i in range(len(d1) - 1):
                new_entry.append(d1[i])
            for i in range(len(d2) - 1):
                new_entry.append(d2[i])
            new_entry.append(track_label)

            new_entries.append(new_entry)

    new_data['data'] = new_entries


    ## Writeback

    destination = ''.join((destination_path, '\\', new_data['relation'], '.arff'))
    out_file = open(destination, 'w+')
    arff.dump(new_data, out_file)





