import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import sys
import io

# Set paths to your dataset
train_dir = r'C:\Users\HP\Desktop\cats and dogs AI classifier\cats and dogs dataset\train'
validation_dir = r'C:\Users\HP\Desktop\cats and dogs AI classifier\cats and dogs dataset\test'

# Redirect stdout to handle encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescaling for validation
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True  # Shuffle the data for better training
)

# Validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # No need to shuffle validation data
)

# Load your saved model
model = load_model(r'C:\Users\HP\Desktop\cats and dogs AI classifier\cats_and_dogs_classifier.h5')

# Function to classify a new image and show prediction confidence
def classify_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch axis

    # Normalize the image
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    # Determine class and confidence
    if confidence > 0.5:
        class_label = 'Dog'
        confidence_percentage = confidence * 100
    else:
        class_label = 'Cat'
        confidence_percentage = (1 - confidence) * 100

    return class_label, confidence_percentage

# Example usage of classify_image function
image_path = r'C:\Users\HP\Desktop\cats and dogs AI classifier\a_mutation_between_a_dog_and_cat.jpeg'  # Update with your image path
result_label, confidence = classify_image(image_path)
print(f'The image is classified as: {result_label} with a confidence of {confidence:.2f}%')

# Function to plot prediction confidence
def plot_prediction_confidence(label, confidence):
    plt.figure(figsize=(6, 4))
    
    # Determine confidence values for plotting
    if label == 'Dog':
        values = [confidence, 100 - confidence]
    else:
        values = [100 - confidence, confidence]

    categories = ['Dog', 'Cat']
    
    plt.bar(categories, values, color=['blue', 'orange'])
    plt.title(f'Prediction Confidence: {label}')
    plt.xlabel('Class')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 100)
    
    # Annotate bars with their values
    for i, v in enumerate(values):
        plt.text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom')

    plt.show()

# Plot the prediction confidence
plot_prediction_confidence(result_label, confidence)
