import sys
from pathlib import Path
import FileManager
from ClassificationModel import ClassificationModel
import argparse
import ProgressIndicator

parser = argparse.ArgumentParser("Zoa Image Classifier", description="Basic tool to train and classify images of Zoanthid coral")
parser.add_argument("--train", type=Path, help="The path of the folder containing the training data")
parser.add_argument("--classify", type=Path, help="The path of an image to classify")
parser.add_argument("saved_model", type=Path, help="The path of the saved model to be used for classification, if no model exists one will be saved here")
args = parser.parse_args()

if args.train is not None:
    # Train the machine learning model with the training data
    ProgressIndicator.loading('Training model...')
    model = ClassificationModel()
    print(model.image_size)
    image_data, image_labels = FileManager.get_training_data(path=args.train, image_size=model.image_size)
    score = model.train(data=image_data, labels=image_labels)
    FileManager.save_model(model=model, path=args.saved_model)
    ProgressIndicator.done()
    print(f'Model accuracy: {score:.2%}\n')


if args.classify is not None:
    # Use the trained model to classify one of the sample test images
    ProgressIndicator.loading('Classifying image...')
    model = FileManager.get_model(path=args.saved_model)
    if model is None: sys.exit('No saved model found')
    image_data = FileManager.get_image_data(path=args.classify, size=model.image_size)
    result = model.classify(image_data=image_data)
    ProgressIndicator.done()
    print(f'Class probabilities for: {args.classify}')
    for prediction in result: print(prediction)
