# Genre Classification

## Purpose

The purpose of this project was to create classifiers to categorize songs by their respective genres. Manually categorizing songs by genre can be a time-intensive and subjective process especially with the frequent invention of new genres. Machine Learning appears to be applicable for genre assignment as it is merely a classification problem, where each category is already known. Automated genre assignment is a popular and broadly applied technology in most popular music collections such as Spotify and Apple Music.


## Million Song Dataset

The Million Song Dataset is a collection of songs that have been tagged with various labels, including genres, and along with datasets of features that have been extracted from the MP3 files of those songs. This dataset was specifically chosen as it offers an extremely large training dataset that is excellent for machine learning models and will hopefully decrease the chances of any of our models overfitting the data. In addition, the creators of the dataset, Echo Nest and LabROSA, included numerous small Python scripts that separated out the datasets into training and validation sets as well as combined the features with their specific label. We were therefore able to jump straight into testing and training various models instead of having to worry about all the intricacies of loading and formatting data.
  
The specific dataset we used from the Million Song Dataset was the Top-MAGD Million Song Dataset Benchmarks. This dataset was created from the Million Song Dataset by the Vienna University of Technology. From this dataset we used the partition mapping files to split the feature set into testing and training data, the label assignment file, and various feature files that will be detailed in the next section. Most importantly, we used the split files that split the dataset into a training set of 2,000 songs from each genre and a testing set that was the rest of the songs. The original dataset consisted of roughly 350,000 songs but more than 200,000 of them were all from the ‘Pop_Rock’ category. This had a very strong chance of leading any of our models to classify all songs as that genre due to the imbalance in the data, so we chose a stratified split that would ensure the training data had an equal number of every song.

The 13 genre labels are Blues, Country, Electronic, Folk, International, Jazz, Latin, New Age, Pop Rock, Rap, Reggae, RnB, and Vocal.
    
## Methods

All of the models were created in Python3.7 using the library Keras with the TensorFlow backend. Keras was chosen as it allows for the very simple creation of complex Neural Network models and eliminates the need to work out all of the math for the number of weights and input and output sizes for each layer. This made quickly changing the architecture of various models quick and easy and allowed for time to be spent on training and analyzing how models performed.

TensorFlow was chosen as the backend for Keras as it cooperated well with Ubuntu and Nvidia drivers which then allowed for the use of a GPU to increase the speed of training.

All of the models shown in this document were trained and analyzed on a computer with an RTX 2060 with 6GB of GDDR6 memory, 16GB of DDR4 memory, and an Intel core i5-9600KF processor.

In common among all of the models was the output layer and loss function. The output layer consisted of a 13 node softmax activated perceptron layer. There were 13 nodes for the 13 possible genre labels and softmax was used as we were training against a one hot vector that represented the ground truth genre label. The loss functions was Keras’s categorical cross entropy function. This was used as the problem is a categorizational problem and only a single label is applicable to any data point. This is not necessarily true for a real world application, but our data assigns a single label to each song and there is no way to apply a loss function for our model saying a song is a mix of Jazz and New Age music.

For each model below, hyperparameters were tuned manually until the model was seen to be overfitting the data. Specific parameters included the number of epochs, layer and kernel sizes, and number of layers. In general, hyperparameters were tuned such that the validation accuracy would be maximized.

### Perceptrons and Neural Networks

Basic neural network models were used on the Marsyas and Rhythm Histogram feature sets. Neither of these feature sets had a time component nor had an extremely large number of features. Therefore, it was decided that a simple neural network would suffice to classify songs using these features.

#### Rhythm Histogram

Magnitudes of modulation frequency are summed over all 24 frequency bands, creating bins of the total amount of modulation at different energy levels.
Creates bins of generally how high and low energy the rhythms are.
Images from website

## Architecture



#### Marsyas 

Identifies and categorizes different 'types' of sounds in the music

## Architecture


### Convolutional Neural Networks

A convolutional neural network was used for all of the other feature sets for the main reason of extracting relevant features while minimizing computational power. For example, the Rhythm Pattern feature set had a total of 1,440 features for each song. A neural network would need many thousands of nodes to make sense of these features meaning millions of weights which would be infeasible to train given our memory and computational constraints. Using a number of convolutional layers, the feature space was shrunk enough to allow for a reasonably sized neural network to analyze the resulting features.

In addition, many of these datasets included a temporal dimension that could be simplified using a kernel. Convolutional layers are known to be useful when analyzing data across a temporal dimension and we took advantage of that when presented with features spread across a temporal dimension.

#### Rhythm Pattern 

Describes sound modulation across 24 frequency bands.
Images of rhythm pattern spectrograms

## Architecture


#### Statistical Spectrum Descriptor



## Architecture



#### Temporal Statistical Spectrum Descriptor

Spectrograms are created with the same method of Rhythm patterns, but covering different time sections throughout the song. Statistical measures are collected over each time step and compiled into a 24x7x7 array of features.
Describes changes in rhythm over time using statistical measures of multiple spectrograms.

   
  


## Results:

### Neural Networks

JMIR Low Level Spectral Data  

Rhythm Histograms                            	Accuracy: 32.1%
						Loss:	2.24
              
        
Marsyas Timbral Data                       	Accuracy: 70.1%
						Loss:	.963




### Convolutional Neural Networks

Rhythm Patterns				Accuracy: 31%
						Loss: 3.8

Statistical Spectrum Data			Accuracy 46%
						Loss 1.8

Temporal SSD				Accuracy 46%
						Loss 1.8



Given the above results, it is clear that the Marsyas Timbral data are the best features to use when classifying music genres. Many of the models would fit the training data well but struggle to get anywhere with the validation data. Only the Marsyas features were able to sufficiently split the feature space into genre categories such that the boundaries worked well in both the training and validation phases. It appears that the other feature sets were not able to sufficiently partition the feature space or lacked the necessary information to discriminate between various genres.

One reason that the Marsyas features were able to classify music genre so well is that the features are composed of heavily processed information extracted from the songs. Marsyas (Music Analysis, Retrieval and SYnthesis for Audio Signals) is an audio processing library that focuses on extracting information from various signal sources and focusing on music. It is unclear how exactly the Vienna University of Technology used the library to extract information as the link is dead, but, given the above results, it seems that the Marsyas features are heavily correlated with the genre of the song and there is only minimal extra analysis needed to extract that correlation.

## Future Work

The Marsyas featureset worked quite well for classifying music but lacks in its ability to classify folk music. A future project could involve using an ensemble classifier that took advantage of the Marsyas feature sets overall ability to classify music along with the Rhythm Pattern model’s ability to classify folk music to create a better overall classifier.
