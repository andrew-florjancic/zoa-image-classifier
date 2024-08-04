import os
from skimage.io import imread
from skimage.transform import resize
import numpy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

training_path = os.path.join(__file__, '..\sample-images\\training-images')

data = []
labels = []
# Fetch the image data from the training folder
for folder in os.listdir(training_path):
    for file in os.listdir(os.path.join(training_path, folder)):
        image_path = os.path.join(training_path, folder, file)
        image = imread(image_path)
        resized_image = resize(image, (100, 100))
        image_data = resized_image.flatten()
        data.append(image_data)
        labels.append(folder)

image_data = numpy.asarray(data)
image_labels = numpy.asarray(labels)

# Reduce dimensions of training and test data with Principal Component Analysis
pca = PCA(n_components=10)
image_data = pca.fit_transform(image_data)

# Split the data set for training and validating the model
x_train, x_test, y_train, y_test = train_test_split(image_data, image_labels, test_size=0.2, shuffle=True,
                                                    stratify=image_labels)

# Tune hyperparameters using grid search then train the model
# Let's just test out some different parameters and see what happens
svc_parameters = [{'estimator__gamma': [0.01, 0.001, 0.0001],
                   'estimator__C': [1, 10, 100, 1000],
                   'estimator__kernel': ['poly'],
                   'estimator__degree': [1, 3, 5, 7, 10]}]

grid_search = GridSearchCV(OneVsRestClassifier(SVC(probability=True)), param_grid=svc_parameters)
grid_search.fit(x_train, y_train)
model = grid_search.best_estimator_

# Test performance of the best model after tuning hyperparameters
y_prediction = model.predict(x_test)
score = accuracy_score(y_test, y_prediction)
print(f'Accuracy: {score:.2%}')

test_path = training_path = os.path.join(__file__, '..\sample-images\\test-images')

# Classify each of the sample test images
for file in os.listdir(training_path):
    image = imread(os.path.join(training_path, file))
    resized_image = resize(image, (100, 100))
    image_data = resized_image.flatten()
    image_data = numpy.asarray(image_data).reshape(1, -1)
    image_data = pca.transform(image_data)
    prediction = model.predict(image_data)
    print(f'{file}:{prediction}')
