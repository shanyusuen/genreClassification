# genreClassification
Repo #2 for CS4641  
Task: Classifying musical genres using data extracted from mp3s.

## The Data

Million Song Dataset:

The Million Song Dataset 
    -A freely-available collection of audio features and metadata for a million contemporary popular music tracks. 
    -Collaborative project started by Echo Nest and LabROSA
    
Top-MAGD Million Song Dataset Benchmarks
    -From Information and Software Engineering Group at TU Wien in Vienna
    -Extracted features of mp3s for use in classification
    
Features being used in classification task:
    -Rythm Patterns
        -Describes sound modulation across 24 frequency bands.
        -images of rythm pattern spectrograms
    -Rythm Histograms
        -Magnitudes of modulation frequency are summed over all 24 frequency bands, creating bins of the total amount of modulation at different energy levels.
        -Creates bins of generally how high and low energy the rythms are.
        -images from website
    -Temporal Statistical Spectrum Descriptor
        -Spectrograms are created with the same method of Rythm patters, but covering different time sections throught the song. Statistical measures are collected over each time step and compiled into a 24x7x7 array of features.
        -Describes changes in rythm over time using statistical measures of multiple spectrograms.
    -Temporal Rythm Histograms
        -include this one?
    -Modulation Frequency Variation Descriptor
        -include this one?
        
    -Marsyas Timbral Data
        -Identifies and categorizes differnt 'types' of sounds in the music
    
    -JMIR: jAudio package for MIR (music information retrieval)
        -A audio processing library equipped to extract a large variety of information from music files.    
    -JMIR low level spectral features
        -(Spectral Centroid, Spectral Rolloff Point, Spectral Flux, Compactness, Spectral Variability, Root Mean Square, Zero Crossings, and Fraction of Low Energy Windows)
        -Describes standard statistic data from sound spectrograms 
    -JMIR MFCC features
        -MFCC: Mel Frequency Cepstral Coefficient
        -Cryptograchic process used to isolate human voice data
    
## Methods

Perceptrons and Neural Networks

Convolutional Neural Networks
     
     
    
background



##Results:

JMIR Low Level Spectral Data                Accuracy: 69.2%
    -TopMAGD labels, Stratified Testing split           
Marsyas Timbral data                        Accuracy 71%
Rythm Histograms                            Accuracy 68%




##Using the Classifier

attach labels to data and split into test and data files using create_partitions.py
for example:

create_partitions.py -o out -f MSD_JMIR_SPECTRAL_ALL_All.arff -l labelsTopMAGD -s splitsTopMAGD

-f denotes the .arff file containing data features
-l denotes the file mapping the data entries to the correct labels
-s denotes whether a data entry will fall into testing or training data
-o is the directory where the result will be stored

all of these files can be downloaded from http://www.ifs.tuwien.ac.at/mir/msd/download.html



run perceptron.py to train a classifier and test.



It is possible to run into different problems do to different .arff feature files being formatted differently.
Data entries should not have trailing commas, and genre labels should be one word or in quotes

dataRead.py contains a script to remove spaces from genres