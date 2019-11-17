#from scipy.io import arff
#import pandas as pd

"""
#Rythm patterns from Million song top magd
data = arff.loadarff('msd-rp.arff')
df = pd.DataFrame(data[0])
"""

#from liac-arff
import arff
import pprint
import codecs
import numpy as np


def load_data(arff_file):
    fp = open(arff_file)
    dataset = arff.load(fp)
    data = np.array(list(dataset['data']))
    return data



def remove_spaces_from_labels(file):
    old = open(file, 'r')
    lines = []
    for line in old:
        line.replace('New Age', 'New_Age')
        lines.append(line.replace('New Age', 'New_Age'))
    open(file, 'w').write(''.join(lines))

def remove_trailing_commas(file):
    old = open(file, 'r')
    lines = []
    for line in old:
        line = line.rstrip(',\n')
        line += '\n'
        lines.append(line)
    open(file, 'w').write(''.join(lines))

#remove_spaces_from_labels('labelsTopMAGD')
#remove_trailing_commas('msd-rh.arff')

"""
if __name__ == "__main__":
    data_file = ".\\out\\msd-rh_msd-topMAGD-genreAssignment.arff"
    d = load_data(data_file)
    print(d.shape)

"""



