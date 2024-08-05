#  Copyright (c) 2024. Andrew Florjancic

"""
Basic file manager module to get training and test data.
"""

import os
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
import numpy
from ClassificationModel import ClassificationModel
import joblib


def get_training_data(path: Path, image_size: int = None) -> (numpy.ndarray[float], numpy.ndarray[str]):
    """
    Get the images located in the provided directory and return the image data and labels.
    :param path: The path where the training data is located.
    :param image_size: Resize all images to the new size if one is provided.
    :return: A tuple containing flattened image data and corresponding classification labels
    """
    data = []
    labels = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            image_path = Path.joinpath(path, folder, file)
            image_data = get_image_data(path=image_path, size=image_size)
            data.append(image_data)
            labels.append(folder)
    return numpy.asarray(data), numpy.asarray(labels)


def get_image_data(path: Path, size: int = None) -> numpy.ndarray[float]:
    """
    Get the image from the provided path and return the image data as a flattened array.
    :param path: The path where the image is located.
    :param size: Resize the image to the new size if one is provided.
    :return: An array of flattened image data.
    """
    image = imread(path)
    if size is not None:
        image = resize(image, (size, size))
    return image.flatten()


def get_model(path: Path) -> ClassificationModel:
    """
    Gets the saved model from the provided location.
    :param path: The location of the saved model.
    :return: The saved ClassificationModel object.
    """
    return joblib.load(filename=path)


def save_model(model: ClassificationModel, path: Path):
    """
    Saves the model to the provided location.
    :param model: The ClassificationModel object to save.
    :param path:  The location to the saved model.
    """
    joblib.dump(value=model, filename=path)


