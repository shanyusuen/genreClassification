# genreClassification
Repo #2 for CS4641


classification steps:


attach labels to data and split into test and data files using create_partitions.py
for example:

create_partitions.py -o output -f MSD_JMIR_SPECTRAL_ALL_All.arff -l LabelsTopMAGD -s TrainingSplitsTopMAGD

-f denotes the .arff file containing data features
-l denotes the file mapping the data entries to the correct labels
-s denotes whether a data entry will fall into testing or training data
-o is the directory where the result will be stored

all of these file can be downloaded from http://www.ifs.tuwien.ac.at/mir/msd/download.html



run perceptron.py to train a classifier and test.



It is possible to run into different problems do to different .arff feature files being formatted differently.
Data entries should not have trailing commas, and genre labels should be one word or in quotes

dataRead.py contains a script to remove spaces from genres