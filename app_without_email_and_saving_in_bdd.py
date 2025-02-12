import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
from flask import Flask, render_template, request, Response
import os
from datetime import datetime

app = Flask(__name__)

# Charger le modèle entraîné
model = load_model('optimized_face_recognition_model.h5')

# Charger les labels des classes
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Définir les labels criminels
criminal_labels = ["elizabeth_holmes", "edward_snowden"]

# Définir un seuil de confiance
confidence_threshold = 0.95

# Charger le cascade Haar pour la détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paramètres de détection des visages
scaleFactor = 1.6
minNeighbors = 7
min_face_size = (30, 30)

# Variable pour le contrôle du flux
streaming = True
UPLOAD_DIR = 'uploads'

def generate_frames(video_path):
    """Générer les frames à partir d'une vidéo uploadée."""
    cap = cv2.VideoCapture(video_path)

    while streaming:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=min_face_size)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100)).astype('float32') / 255.0
            face_resized = face_resized.reshape(-1, 100, 100, 1)

            prediction = model.predict(face_resized)
            predicted_class = np.argmax(prediction)
            predicted_confidence = prediction[0][predicted_class]

            if predicted_confidence >= confidence_threshold:
                predicted_label = class_labels[predicted_class]
                if predicted_label in criminal_labels:
                    label_text = f"Criminal: {predicted_label}"
                    color = (0, 0, 255)  # Rouge pour les criminels
                else:
                    label_text = "Not Criminal"
                    color = (0, 255, 0)  # Vert pour non-criminels
            else:
                label_text = "Not Criminal"
                color = (0, 255, 0)  # Vert pour inconnu

            # Dessiner le rectangle et le label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convertir l'image en format JPEG pour la diffusion
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def process_image(image_path):
    """Process a single image for prediction."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at path: {image_path}")
        return None, None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (100, 100))
    img_resized = img_resized.astype('float32') / 255.0
    img_reshaped = img_resized.reshape(-1, 100, 100, 1)

    prediction = model.predict(img_reshaped)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    return predicted_label, img

def generate_frames_from_camera():
    """Generate frames from the live camera feed."""
    cap = cv2.VideoCapture(0)

    while streaming:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=min_face_size)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100)).astype('float32') / 255.0
            face_resized = face_resized.reshape(-1, 100, 100, 1)

            prediction = model.predict(face_resized)
            predicted_class = np.argmax(prediction)
            predicted_confidence = prediction[0][predicted_class]

            if predicted_confidence >= confidence_threshold:
                predicted_label = class_labels[predicted_class]
                if predicted_label in criminal_labels:
                    label_text = f"Criminal: {predicted_label}"
                    color = (0, 0, 255)  # Red for criminals
                else:
                    label_text = "Not Criminal"
                    color = (0, 255, 0)  # Green for non-criminals
            else:
                label_text = "Not Criminal"
                color = (0, 255, 0)  # Green for unknown

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert frame to JPEG format for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """Page d'accueil."""
    return render_template('index.html')  # Remplacez 'index.html' par votre propre template si nécessaire


@app.route('/predict_camera')
def predict_camera():
    """Prédiction via la caméra en direct."""
    global streaming
    streaming = True  # Activer le flux pour la caméra
    return Response(
        generate_frames_from_camera(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Prédiction sur une vidéo uploadée."""
    global streaming
    streaming = True  # Réinitialiser le flux pour une nouvelle vidéo

    if 'video' not in request.files:
        return "No video uploaded", 400

    video = request.files['video']
    video_path = f"uploads/{video.filename}"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)  # S'assurer que le répertoire existe
    video.save(video_path)

    return Response(
        generate_frames(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Prédiction sur une image uploadée."""
    try:
        # Vérifiez si une image est incluse dans la requête
        if 'image' not in request.files:
            return "No image uploaded", 400

        # Sauvegarde de l'image envoyée
        image_file = request.files['image']
        image_path = f"uploads/{image_file.filename}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)  # S'assurer que le répertoire existe
        image_file.save(image_path)

        # Processus de prédiction pour l'image
        prediction_label, processed_image = process_image(image_path)

        # Si l'image ne peut pas être traitée, retournez une erreur
        if processed_image is None:
            return f"Error processing image at path: {image_path}", 400

        # Sauvegarder l'image traitée pour l'affichage
        result_image_path = os.path.join('static', 'uploads', image_file.filename)
        os.makedirs(os.path.dirname(result_image_path), exist_ok=True)  # S'assurer que le répertoire existe
        cv2.imwrite(result_image_path, processed_image)

        # Liste noire de criminels (à adapter selon vos besoins)
        blacklist = ['elizabeth_holmes', 'edward_snowden']  # Remplacez par vos propres noms
        if prediction_label in blacklist:
            print(f"Criminal detected: {prediction_label}")

        # Retourner la page de résultat avec le label de prédiction
        return render_template(
            'result.html',
            results=[prediction_label],
            image_path=result_image_path
        )

    except Exception as e:
        # Gérer les erreurs générales et retourner un message d'erreur
        return f"An error occurred: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
