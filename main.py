import os
import FileManager
from ClassificationModel import ClassificationModel

# Train the machine learning model with the training data
training_path = os.path.join(__file__, '..\sample-images\\training-images')
model = ClassificationModel()
image_data, image_labels = FileManager.get_training_data(path=training_path, image_size=model.image_size)
score = model.train(data=image_data, labels=image_labels)
print(f'Model accuracy: {score:.2%}\n')

# Use the trained model to classify one of the sample test images
test_path = os.path.join(__file__, '..\sample-images\\test-images\\sour_lemons.jpeg')
image_data = FileManager.get_image_data(path=test_path, size=model.image_size)
result = model.classify(image_data=image_data)
print(f'Class probabilities for: {test_path}')
for prediction in result: print(prediction)
