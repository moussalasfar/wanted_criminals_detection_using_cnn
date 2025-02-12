import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import seaborn as sns

def load_and_preprocess_images(base_path):
    images = []
    labels = []
    classes = os.listdir(base_path)

    for class_label in classes:
        class_path = os.path.join(base_path, class_label)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            # Load the image
            img = cv2.imread(image_path)
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize the image
            img_resized = cv2.resize(img_gray, (100, 100))
            images.append(img_resized)
            labels.append(class_label)

    return np.array(images), np.array(labels)

# Load training data
train_images, train_labels = load_and_preprocess_images('train/')
# Load validation data
validation_images, validation_labels = load_and_preprocess_images('test/')

def visualize_sample_images(images, labels, num_images=40):
    plt.figure(figsize=(20, 20))  # Adjust the figure size for better visibility
    for i in range(num_images):
        plt.subplot(8, 5, i + 1)  # Create a 10x10 grid for 100 images
        plt.imshow(images[i], cmap='gray')  # Display image in grayscale
        plt.title(labels[i])
        plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()

# Visualize 40 sample training images
visualize_sample_images(train_images, train_labels, num_images=40)

# Normalize images
train_images = train_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0

# Convert labels to categorical
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
validation_labels_encoded = label_encoder.transform(validation_labels)

# Since we are using multi-class classification (criminal detection with multiple criminals),
# make sure to encode labels for each criminal and "not criminal"
train_labels_categorical = to_categorical(train_labels_encoded, num_classes=3)  
validation_labels_categorical = to_categorical(validation_labels_encoded, num_classes=3)

# Check if the labels are one-hot encoded correctly
print("Train labels shape:", train_labels_categorical.shape)
print("Validation labels shape:", validation_labels_categorical.shape)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Regularization to prevent overfitting
    Dense(3, activation='softmax')  # Output layer with 3 classes: "angelina jolie", "will smith", "not criminal"
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images.reshape(-1, 100, 100, 1),  # Reshape for grayscale images
    train_labels_categorical,
    validation_data=(validation_images.reshape(-1, 100, 100, 1), validation_labels_categorical),
    epochs=70,
    batch_size=32
)

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.show()

plot_training_history(history)

# Save the trained model
model.save('face_recognition_model.h5')

# Save the class labels to a file
class_labels = label_encoder.classes_.tolist()
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)

# Evaluate the model on the validation set and compute metrics
validation_predictions = model.predict(validation_images.reshape(-1, 100, 100, 1))
# Convert one-hot encoded predictions back to labels
validation_predictions_labels = np.argmax(validation_predictions, axis=1)

# Generate classification report with precision, recall, and F1 score
print("Classification Report:\n", classification_report(validation_labels_encoded, validation_predictions_labels, target_names=label_encoder.classes_))

# Generate confusion matrix
conf_matrix = confusion_matrix(validation_labels_encoded, validation_predictions_labels)
print("Confusion Matrix:\n", conf_matrix)

# Répartition des classes dans l'ensemble d'entraînement
train_class_counts = Counter(train_labels)

# Répartition des classes dans l'ensemble de validation
validation_class_counts = Counter(validation_labels)

# Diagramme des classes pour l'ensemble d'entraînement
plt.figure(figsize=(12, 6))
plt.bar(train_class_counts.keys(), train_class_counts.values(), color='skyblue', alpha=0.7)
plt.title("Nombre d'images par classe dans l'ensemble d'entraînement")
plt.xlabel("Classes")
plt.ylabel("Nombre d'images")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Diagramme des classes pour l'ensemble de validation
plt.figure(figsize=(12, 6))
plt.bar(validation_class_counts.keys(), validation_class_counts.values(), color='orange', alpha=0.7)
plt.title("Nombre d'images par classe dans l'ensemble de validation")
plt.xlabel("Classes")
plt.ylabel("Nombre d'images")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Classes prédites")
plt.ylabel("Classes réelles")
plt.title("Matrice de confusion")
plt.show()

