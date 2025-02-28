# zoa-image-classifier
Zoanthids, commonly referred to as Zoas, are a popular type of soft coral in the saltwater aquarium industry that come in a wide variety of colors and patterns. The industry collectively assigns fun and silly common names to popular strains for easier communication and marketing hype.

It can be difficult to correctly identify an unknown Zoa because there are so many different variations that exist, and some look very similar to each other with subtle differences. Also, good luck if you're like me and have some form of color vision deficiency. Classifying Zoas seems like a suitable candidate for a multi-class image classifier.

## Design
This is just a fun small proof of concept project and is not intended to be used in any kind of production environment or commercial application. The plan is to create a tool to, hopefully, correctly classify an image of a Zoanthid coral by its common name. Anything that can't realistically be completed while working at a leisurely pace over a couple days and doesn't make my brain melt is out of scope for this project. That loosely translates to only collecting a small sample size of data, no style guide adoption, minimal error handling, no unit tests, no CI/CD pipeline, and no custom implementations of machine learning algorithms. Below are some project details and design decisions that will affect the performance and accuracy of this classification tool:
  - The classes included in the sample training data set are a very small subset of the total number of classes that actually exist. I expect that adding more classes to the data set will lower the accuracy. This isn't necessarily the end of the world for a tool like this, it would still be useful to provide a list of most likely candidates.
  - I'm already familiar with the scikit-learn Python machine learning library so I'll just use this one, but TensorFlow or Pytorch would probably offer better performance because they both support computations on a GPU.
  - A convolutional neural network would probably work best for image classification, but it doesn't look like scikit-learn has one, so I'll settle for one of the support vector machine classifiers included in the library.
  - The sample size of data is small and only collected from one source aquarium which will negatively affect the classification accuracy especially if attempting to classify a coral from a completely different source than the training data. Unfortunately, data collection is time consuming so collecting more samples is out of scope for this project. There are so many different environmental factors that can also affect coral appearance so collecting samples from many different sources would likely improve the usability of this tool for a wider range of environments. 
  - The image quality of data collected is poor which will likely result in an increased misclassification rate. With the current method of data collection there are a lot of key characteristics that may be missed or partially like number of tentacles, oral disk size, mouth shape and size, etc. Coral photography with an old iPhone SE is limited to the capability of the phone's CMOS sensor which doesn't seem to handle the blue spectrum of light very well. A DSLR with a full frame sensor and a macro lens would produce far superior images which should help improve classification accuracy. In addition to the improved hardware, it would also be beneficial to standardize the image capturing procedure to ensure all pictures are taken under a similar spectrum of light and at an equal distance from the camera.

## Required Packages
- numpy
- scikit-learn
- scikit-image

## Usage
```
usage: Zoa Image Classifier [-h] [--train TRAIN] [--classify CLASSIFY] saved_model

Basic tool to train and classify images of Zoanthid coral

positional arguments:
  saved_model          The path of the saved model to be used for classification, if no model exists one will be saved
                       here

options:
  -h, --help           show this help message and exit
  --train TRAIN        The path of the folder containing the training data
  --classify CLASSIFY  The path of an image to classify
```

## Example
```
>python - u main.py --train sample-images\training-images --classify sample-images\test-images\rastas.jpeg saved_model.joblib 
Training model... ✅
Model accuracy: 100.00%

Classifying image... ✅
Class probabilities for: sample-images\test-images\rastas.jpeg
 64.44%: rastas
 18.64%: daisy_cutter
 15.36%: king_midas
  0.90%: sour_lemons
  0.59%: utter_chaos
  0.07%: daisy_dukes
```
