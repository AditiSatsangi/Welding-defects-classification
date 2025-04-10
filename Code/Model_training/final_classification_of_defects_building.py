# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

"""## Importing Liabraries"""

!pip install tensorflow opencv-python matplotlib scikit-learn

"""## Data Collection"""

import os
import cv2
import numpy as np
import pandas as pd

# Initialize lists to hold the images and labels
images = []
labels = []

import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

# Define your paths
original_images_path = '/content/drive/MyDrive/Building_Defects/Combined_Equally_Images'
categories = [ 'Overweld', 'Porosity','Undercut', 'Underfilled']
label_map = {0: 'Overweld', 1: 'Porosity', 2: 'Undercut', 3: 'Underfilled'}

# Initialize lists to hold the original and augmented images and labels
original_images = []
original_labels = []

# Load original images and labels
for category in categories:
    category_path = os.path.join(original_images_path, category)
    label = list(label_map.keys())[list(label_map.values()).index(category)]

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize to match model input
            original_images.append(img)
            original_labels.append(label)

# Convert to numpy arrays
combined_images = np.array(original_images)
combined_labels = np.array(original_labels)

# Print combined dataset size
print(f"Combined dataset size: {len(combined_images)} images")
print(f"Combined labels size: {len(combined_labels)} labels")

# Convert to numpy arrays
image = np.array(combined_images)
label = np.array(combined_labels)

# Create a DataFrame for easy manipulation
data = pd.DataFrame({'image': list(image), 'label': label})

from matplotlib import pyplot as plt
categories = ['Overweld', 'Porosity', 'Undercut', 'Underfilled']
label_map = {0: 'Overweld', 1: 'Porosity', 2: 'Undercut', 3: 'Underfilled'}

# Plot the histogram
data['label'].plot(kind='hist', bins=20,title='Building Defects')
plt.gca().spines[['top', 'right']].set_visible(False)

# Set the x-axis tick labels to category names
plt.xticks(ticks=range(len(categories)), labels=categories)

# Show the plot
plt.show()

# Count occurrences of each label
label_counts = data['label'].value_counts()
print(label_counts)
print("------------------")
print(label_map)

combined_images.shape

combined_labels.shape

print(len(combined_images), len(combined_labels))

"""## Data Analysis"""

import matplotlib.pyplot as plt

# Function to display images
def display_images(images, labels, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].astype('uint8'))  # Convert to uint8 for display
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

# Display sample images from the combined dataset
display_images(combined_images, combined_labels, num_images=4)

combined_labels

"""## Split the Data"""

from sklearn.model_selection import train_test_split

# Split combined dataset into training (70%), validation (20%), and test (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(combined_images, combined_labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Print sizes of the splits
print(f"Training dataset size: {len(X_train)} images")
print(f"Validation dataset size: {len(X_val)} images")
print(f"Test dataset size: {len(X_test)} images")

print(f"Train/Val size: {len(X_train_val)}, Test size: {len(X_test)}")
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

print(X_train[0])

"""## Data Preprocessing"""

# Normalize images
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

import matplotlib.pyplot as plt

# Display a few training images
for i in range(5):
    plt.imshow(X_train[i])
    plt.title(f"Label: {y_train[i]}")
    plt.show()

"""## Model Training

## VGG 16
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# One-hot encode the labels
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Load the VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base to use pre-trained features
for layer in base_model.layers:
    layer.trainable = False

# Add new layers for your task
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)  # Fully connected layer
x = Dropout(0.4)(x)  # Dropout for regularization
predictions = Dense(4, activation='softmax')(x)  # 4 categories for defects

# Define the complete model
vgg_model = Model(inputs=base_model.input, outputs=predictions)

vgg_model.compile(optimizer= tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Display the model architecture
vgg_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = vgg_model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=40,
                        batch_size=32,
                        callbacks=[early_stopping])

# Save the model after training
vgg_model.save('vgg_defect_classifier.h5')

from tensorflow.keras.utils import plot_model
# Generate a plot of the model architecture
plot_model(
    vgg_model,
    to_file='model_architecture.png',  # Save the plot to a file
    show_shapes=True,                  # Display the shape of the layers
    show_layer_names=True,             # Display layer names
    dpi=96                             # Set the resolution of the image
)



"""### Accuracy"""

# Evaluate the model on the test set
test_loss, test_acc = vgg_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('VGG-16 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VGG-16 Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make predictions on the test set
predictions = vgg_model.predict(X_test)

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Convert predictions to class indices (argmax to get the predicted class)
predicted_labels = np.argmax(predictions, axis=1)

# 2. Convert y_test (one-hot encoded) to class indices
y_test_labels = np.argmax(y_test, axis=1)

# 3. Calculate the confusion matrix
confusion_matrix1 = tf.math.confusion_matrix(y_test_labels, predicted_labels)

# 4. Convert to numpy for easier handling
confusion_matrix1 = confusion_matrix1.numpy()

# Print the confusion matrix as a raw tensor
print("Confusion Matrix:")
print(confusion_matrix1)
# Optionally, visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix1, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

from sklearn.metrics import classification_report

# Evaluate the model on the test set
test_loss, test_acc = vgg_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")

# Predict class labels
y_pred = np.argmax(vgg_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=categories)  # Replace 'class_names' with your class labels
print("Classification Report:")
print(report)

# Predict class labels
y_pred = np.argmax(vgg_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Calculate precision, recall, and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

"""### Prediction"""

import matplotlib.pyplot as plt

# Function to plot the images with their predicted labels
def plot_predictions(images, true_labels, predicted_labels, categories, num_images=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f"True: {categories[np.argmax(true_labels[i])]}, Pred: {categories[predicted_labels[i]]}")
    plt.tight_layout()
    plt.show()

# Define the categories
categories = ['Overweld', 'Undercut', 'Underfilled', 'Porosity']

#0: Overweld',1:  'Undercut', 2: 'Underfilled', 3: 'Porosity'#
# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Map predicted labels to category names
predicted_categories = [categories[label] for label in predicted_labels]

# Print the predicted categories
print("Predicted categories:", predicted_categories)

plot_predictions(X_test, y_test, predicted_labels, categories, num_images=6)



"""## DenseNet121"""

from tensorflow.keras.applications import DenseNet121  # You can use DenseNet169, DenseNet201, etc.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load the pre-trained DenseNet121 model without the top layer
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers to retain pre-trained features
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for your classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling to reduce dimensions
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5) (x)
predictions = Dense(len(categories), activation='softmax')(x)

# Create the final model
densenet_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
densenet_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the custom layers
history = densenet_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=32)

# Evaluate on the test set
test_loss, test_acc = densenet_model.evaluate(X_test, y_test)
print(f"Test accuracy after fine-tuning: {test_acc}")

# Display model summary
densenet_model.summary()

"""### Accuracy"""

# Save the model after training
densenet_model.save('densenet_defect_classifier.h5')

plot_model(
    densenet_model,
    to_file='densemodel_architecture.png',  # Save the plot to a file
    show_shapes=True,                  # Display the shape of the layers
    show_layer_names=True,             # Display layer names
    dpi=96                             # Set the resolution of the image
)

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('DenseNet Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('DenseNet Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make predictions on the test set
predictions = densenet_model.predict(X_test)

# 1. Convert predictions to class indices (argmax to get the predicted class)
predicted_labels = np.argmax(predictions, axis=1)

# 2. Convert y_test (one-hot encoded) to class indices
y_test_labels = np.argmax(y_test, axis=1)

# 3. Calculate the confusion matrix
confusion_matrix2 = tf.math.confusion_matrix(y_test_labels, predicted_labels)

# 4. Convert to numpy for easier handling
confusion_matrix2= confusion_matrix2.numpy()

# Print the confusion matrix as a raw tensor
print("Confusion Matrix:")
print(confusion_matrix2)
# Optionally, visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix2, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

# Predict class labels
y_pred = np.argmax(densenet_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=categories)  # Replace 'class_names' with your class labels
print("Classification Report:")
print(report)

# Define the categories
categories = ['Overweld', 'Undercut', 'Underfilled', 'Porosity']

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Map predicted labels to category names
predicted_categories = [categories[label] for label in predicted_labels]

# Print the predicted categories
print("Predicted categories:", predicted_categories)



"""### Predictions"""

plot_predictions(X_test, y_test, predicted_labels, categories, num_images=5)

"""## MobileNet"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout

# Load the pre-trained MobileNet model, excluding the top layer
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers for our custom classification task
x = mobilenet_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling instead of flattening
x= Dropout(0.5) (x)
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(len(categories), activation='softmax')(x)  # Output layer for 4 classes

# Define the final model
model = Model(inputs=mobilenet_model.input, outputs=predictions)

# Freeze the layers of MobileNet to avoid retraining them
for layer in mobilenet_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=32)

"""### Accuracy"""

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

plot_model(
    model,
    to_file='mobilemodel_architecture.png',  # Save the plot to a file
    show_shapes=True,                  # Display the shape of the layers
    show_layer_names=True,             # Display layer names
    dpi=96                             # Set the resolution of the image
)

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()

# Save the model after training
mobilenet_model.save('mobilenet_defect_classifier.h5')

"""### Predictions"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict_class(image_path, model, categories):
    """
    Predict the class of an input image using a pre-trained model.

    Args:
    - image_path (str): Path to the input image.
    - model (keras.Model): The trained model for classification.
    - categories (list): List of category names.

    Returns:
    - image: The processed image in the correct shape for display.
    - predicted_class: The name of the predicted category.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the input size expected by the model (e.g., 224x224)
    resized_image = cv2.resize(image_rgb, (224, 224))

    # Preprocess the image for the model (normalization, reshaping, etc.)
    processed_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Use the model to predict the class probabilities
    predictions = model.predict(processed_image)

    # Get the index of the highest probability
    predicted_index = np.argmax(predictions, axis=1)[0]

    # Get the corresponding class name from the categories list
    predicted_class = categories[predicted_index]

    return image_rgb, predicted_class

# Make predictions on the test set
predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

# 2. Convert y_test (one-hot encoded) to class indices
y_test_labels = np.argmax(y_test, axis=1)

# 3. Calculate the confusion matrix
confusion_matrix2 = tf.math.confusion_matrix(y_test_labels, predicted_labels)

# 4. Convert to numpy for easier handling
confusion_matrix2= confusion_matrix2.numpy()

# Print the confusion matrix as a raw tensor
print("Confusion Matrix:")
print(confusion_matrix2)
# Optionally, visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix2, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

# Predict class labels
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=categories)  # Replace 'class_names' with your class labels
print("Classification Report:")
print(report)

# Define the categories
categories = ['Overweld', 'Undercut', 'Underfilled', 'Porosity']

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Map predicted labels to category names
predicted_categories = [categories[label] for label in predicted_labels]

# Print the predicted categories
print("Predicted categories:", predicted_categories)

# Plot some predictions
plot_predictions(X_test, y_test, predicted_labels, categories, num_images=5)

"""## Xception"""

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Start with the base Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model to retain pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers to the base model
x = layers.GlobalAveragePooling2D()(base_model.output)  # Flatten the feature map
x = layers.Dense(1024, activation='relu')(x)  # Add a fully connected layer
x = layers.Dropout(0.9)(x)  # Add dropout for regularization
output_layer = layers.Dense(4, activation='softmax')(x)  # Final output layer for 10 classes

# Create a new model combining the base model and custom head
model_xception = models.Model(inputs=base_model.input, outputs=output_layer)

# Print
model_xception.summary()

plot_model(
    model_xception,
    to_file='Xception_architecture.png',  # Save the plot to a file
    show_shapes=True,                  # Display the shape of the layers
    show_layer_names=True,             # Display layer names
    dpi=96                             # Set the resolution of the image
)

model_xception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model_xception.fit(
    X_train, y_train,  # Training data
    validation_data=(X_val, y_val),  # Validation data
    epochs=40,  # Number of epochs
    batch_size=32  # Batch size
)

"""### Accuracy"""

# Now evaluate with the integer-encoded labels
test_loss, test_acc = model_xception.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

model_xception.summary()

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Xception Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()

plot_model(
    model_xception,
    to_file='xceptionmodel_architecture.png',  # Save the plot to a file
    show_shapes=True,                  # Display the shape of the layers
    show_layer_names=True,             # Display layer names
    dpi=96                             # Set the resolution of the image
)

# Predict class labels
y_pred = np.argmax(model_xception.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

# Classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=categories)  # Replace 'class_names' with your class labels
print("Classification Report:")
print(report)

model.save('xception_defect_classifier1.h5')

"""### Prediction"""

predictions= model_xception.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

# 2. Convert y_test (one-hot encoded) to class indices
y_test_labels = np.argmax(y_test, axis=1)

# 3. Calculate the confusion matrix
confusion_matrix2 = tf.math.confusion_matrix(y_test_labels, predicted_labels)

# 4. Convert to numpy for easier handling
confusion_matrix2= confusion_matrix2.numpy()

# Print the confusion matrix as a raw tensor
print("Confusion Matrix:")
print(confusion_matrix2)
# Optionally, visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix2, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

plot_predictions(X_test, y_test, predicted_labels, categories, num_images=5)

"""## NASNet-Mobile Model"""

from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Load NASNetMobile with pretrained ImageNet weights, without the top layers
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model (NASNetMobile) to prevent them from training
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = base_model.output  # Get the output of the base model
x = GlobalAveragePooling2D()(x)  # Add global average pooling to reduce dimensions
x= Dropout(0.8)(x)
x= Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer with 1024 units and ReLU activation
predictions = Dense(len(categories), activation='softmax')(x)  # Output layer for the number of classes in your dataset

# Create the model
nasmodel = Model(inputs=base_model.input, outputs=predictions)
nasmodel.summary()

# Compile the model
nasmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = nasmodel.fit(
    X_train, y_train,  # Training data
    validation_data=(X_val, y_val),  # Validation data
    epochs=20,  # Number of epochs
    batch_size=32  # Batch size
)

"""### Accuracy"""

# Now evaluate with the integer-encoded labels
test_loss, test_acc = nasmodel.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

nasmodel.save('nasnet_defect_classifier1.h5')

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()

"""### Prediction"""

predictions = nasmodel.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

# 2. Convert y_test (one-hot encoded) to class indices
y_test_labels = np.argmax(y_test, axis=1)

# 3. Calculate the confusion matrix
confusion_matrix2 = tf.math.confusion_matrix(y_test_labels, predicted_labels)

# 4. Convert to numpy for easier handling
confusion_matrix2= confusion_matrix2.numpy()

# Print the confusion matrix as a raw tensor
print("Confusion Matrix:")
print(confusion_matrix2)
# Optionally, visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix2, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Map predicted labels to category names
predicted_categories = [categories[label] for label in predicted_labels]

# Print the predicted categories
print("Predicted categories:", predicted_categories)

# Plot some predictions
plot_predictions(X_test, y_test, predicted_labels, categories, num_images=5)

# Plot some predictions
plot_predictions(X_test, y_test, predicted_labels, categories, num_images=10)





