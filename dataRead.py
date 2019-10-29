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
file_ = codecs.open('msd-rp.arff', 'rb', 'utf-8')
dataset = arff.loads(file_)
pprint.pprint(dataset)
pprint.pprint(dataset['data'])
#data = dataset['data']
decoder = arff.ArffDecoder()
d = decoder.decode(file_, encode_nominal=True)
pprint.pprint(d)


#print(data)