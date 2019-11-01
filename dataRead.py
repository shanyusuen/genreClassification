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

data_file = ".\\out\\msd-rh_msd-topMAGD-genreAssignment.arff"

def load_data(arff_file):
    #file_ = codecs.open(data_file, 'rb', 'utf-8')
    dataset = arff.load(arff_file)
    #data = dataset['data']
    #decoder = arff.ArffDecoder()
    #d = decoder.decode(file_, encode_nominal=True)
    #pprint.pprint(d)
    data = np.array(list(dataset))
    return data


#print(data)

if __name__ == "__main__":
    d = load_data(data_file)
    print(d.shape)