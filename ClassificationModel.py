#  Copyright (c) 2024. Andrew Florjancic

import numpy
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import sys


class Prediction:
    """
    Result of classification containing a label and a probability
    """
    label: str
    probability: float

    def __init__(self, label: str, probability: float):
        """
        :param label: Class name predicted by a model.
        :param probability: Probability that the label is the correct classification.
        """
        self.label = label
        self.probability = probability

    def __str__(self):
        """
        :return: A formatted string containing the probability and label.
        """
        return f'{self.probability:>7.2%}: {self.label}'


class ClassificationModel:
    """
    Classification model used to train and classify images.
    """
    # Model input and hyperparameters
    _image_size = 100
    _test_size = 0.2
    _estimator = SVC(probability=True)
    _estimator_parameters = [{'estimator__gamma': [0.01, 0.001, 0.0001],
                              'estimator__C': [1, 10, 100, 1000],
                              'estimator__kernel': ['poly'],
                              'estimator__degree': [1, 3, 5, 7, 10]}]
    _trained_model: any
    _pca = PCA(n_components=10)

    @property
    def image_size(self):
        """The image size to be used in this classification model."""
        return self._image_size

    def train(self, data: numpy.ndarray[float], labels: numpy.ndarray[str]) -> float:
        """
        Create a new instance of the ML model and train it on the provided data.
        :param data: Flattened image data to train the model.
        :param labels: Corresponding class labels for the data.
        :return: The accuracy score of the trained model.
        """
        # Reduce dimensions of the data with Principal Component Analysis
        data = self._pca.fit_transform(data)

        # Split the data set for training and validating the model
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self._test_size, shuffle=True, stratify=labels)

        # Tune hyperparameters using grid search then train the model
        grid_search = GridSearchCV(OneVsRestClassifier(estimator=self._estimator), param_grid=self._estimator_parameters)
        grid_search.fit(x_train, y_train)
        self._trained_model = grid_search.best_estimator_
        y_prediction = self._trained_model.predict(x_test)
        return accuracy_score(y_test, y_prediction)

    def classify(self, image_data: numpy.ndarray[float]) -> [Prediction]:
        """
        Classify the provided image and return an array of predictions.
        :param image_data: An array of flattened image data for a single image.
        :return: An array of Predictions containing class label and probability.
        """
        if self._trained_model is None: sys.exit('Model needs to be trained first')
        # The model expects 2D array so the single sample needs to be reshaped
        image_data = self._pca.transform(image_data.reshape(1, -1))
        probabilities = self._trained_model.predict_proba(image_data).reshape(-1)
        predictions = list(map(lambda label, probability: Prediction(label, probability), self._trained_model.classes_, probabilities))
        predictions.sort(key=lambda prediction: prediction.probability, reverse=True)
        return predictions
