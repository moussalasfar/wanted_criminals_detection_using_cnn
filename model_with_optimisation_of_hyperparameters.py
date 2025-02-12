import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras_tuner import Hyperband

# Function to load and preprocess images
def load_and_preprocess_images(base_path):
    images = []
    labels = []
    classes = os.listdir(base_path)

    for class_label in classes:
        class_path = os.path.join(base_path, class_label)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (100, 100))
            images.append(img_resized)
            labels.append(class_label)

    return np.array(images), np.array(labels)

# Load and preprocess data
train_images, train_labels = load_and_preprocess_images('train/')
validation_images, validation_labels = load_and_preprocess_images('test/')

# Normalize images
train_images = train_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
validation_labels_encoded = label_encoder.transform(validation_labels)

# One-hot encode labels
train_labels_categorical = to_categorical(train_labels_encoded, num_classes=3)
validation_labels_categorical = to_categorical(validation_labels_encoded, num_classes=3)

# Define the model building function for hyperparameter tuning
def build_model(hp):
    model = Sequential([
        Conv2D(
            hp.Int('filters_layer_1', min_value=32, max_value=128, step=32), 
            (hp.Choice('kernel_size_1', values=[3, 5])), activation='relu', 
            input_shape=(100, 100, 1)
        ),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            hp.Int('filters_layer_2', min_value=32, max_value=128, step=32), 
            (hp.Choice('kernel_size_2', values=[3, 5])), activation='relu'
        ),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            hp.Int('filters_layer_3', min_value=64, max_value=256, step=64), 
            (hp.Choice('kernel_size_3', values=[3, 5])), activation='relu'
        ),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            hp.Int('filters_layer_4', min_value=128, max_value=256, step=64), 
            (hp.Choice('kernel_size_4', values=[3, 5])), activation='relu'
        ),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'),
        Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),
        Dense(3, activation='softmax')  # Output layer
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize the tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    factor=3,
    directory='hyperparam_search',
    project_name='face_recognition_tuning'
)

# Perform the hyperparameter search
tuner.search(
    train_images.reshape(-1, 100, 100, 1),
    train_labels_categorical,
    epochs=30,
    validation_data=(validation_images.reshape(-1, 100, 100, 1), validation_labels_categorical),
    batch_size=32
)

# Get the best hyperparameters
best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(f"Filters Layer 1: {best_hyperparams.get('filters_layer_1')}")
print(f"Filters Layer 2: {best_hyperparams.get('filters_layer_2')}")
print(f"Filters Layer 3: {best_hyperparams.get('filters_layer_3')}")
print(f"Filters Layer 4: {best_hyperparams.get('filters_layer_4')}")
print(f"Kernel Size Layer 1: {best_hyperparams.get('kernel_size_1')}")
print(f"Kernel Size Layer 2: {best_hyperparams.get('kernel_size_2')}")
print(f"Kernel Size Layer 3: {best_hyperparams.get('kernel_size_3')}")
print(f"Kernel Size Layer 4: {best_hyperparams.get('kernel_size_4')}")
print(f"Dropout Rate: {best_hyperparams.get('dropout_rate')}")
print(f"Learning Rate: {best_hyperparams.get('learning_rate')}")
print(f"Dense Units: {best_hyperparams.get('dense_units')}")

# Build the best model
best_model = tuner.hypermodel.build(best_hyperparams)

# Train the best model
history = best_model.fit(
    train_images.reshape(-1, 100, 100, 1),
    train_labels_categorical,
    validation_data=(validation_images.reshape(-1, 100, 100, 1), validation_labels_categorical),
    epochs=70,
    batch_size=32
)

# Save the best model
best_model.save('face_recognition_model.h5')

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

# Evaluate and display confusion matrix
val_loss, val_accuracy = best_model.evaluate(validation_images.reshape(-1, 100, 100, 1), validation_labels_categorical)
train_loss, train_accuracy = best_model.evaluate(train_images.reshape(-1, 100, 100, 1), train_labels_categorical)

print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

# Predict on validation data
predictions = np.argmax(best_model.predict(validation_images.reshape(-1, 100, 100, 1)), axis=1)

# Generate classification report
print("\nClassification Report:")
print(classification_report(validation_labels_encoded, predictions, target_names=label_encoder.classes_))

# Generate confusion matrix
confusion_mat = confusion_matrix(validation_labels_encoded, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
