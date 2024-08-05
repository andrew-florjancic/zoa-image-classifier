import os
import FileManager
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

training_path = os.path.join(__file__, '..\sample-images\\training-images')
image_data, image_labels = FileManager.get_training_data(path=training_path, image_size=100)

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
    image_path = os.path.join(test_path, file)
    image_data = FileManager.get_image_data(path=image_path, size=100)
    # The models are expecting a 2D array so the single sample needs to be reshaped
    reshaped_image_data = image_data.reshape(1, -1)
    transformed_image_data = pca.transform(reshaped_image_data)
    prediction = model.predict(transformed_image_data)
    print(f'{file}:{prediction}')
