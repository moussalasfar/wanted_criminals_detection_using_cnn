import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
from flask import Flask, render_template, request, Response, redirect, url_for
import os
from pymongo import MongoClient
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

# Charger le modèle entraîné
model = load_model('face_recognition_model.h5')

# Charger les labels des classes
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Définir les labels criminels
criminal_labels = ["elizabeth_holmes", "edward_snowden"]

# Définir un seuil de confiance
confidence_threshold = 0.95

# Connexion à MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')  # Remplacez par votre URI MongoDB
    db = client['criminal_detection']  # Nom de votre base de données
    criminals_collection = db['detected_criminals']  # Nom de votre collection
    print("Connexion à MongoDB réussie")
except Exception as e:
    print(f"Erreur de connexion à MongoDB: {str(e)}")

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
    """Générer les frames à partir d'une vidéo uploadée et enregistrer les criminels détectés dans la base de données."""
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

                    # Capture the face from the frame and save it to the uploads directory
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    filename = f'{predicted_label}_{timestamp}.jpg'
                    file_path = os.path.join(UPLOAD_DIR, filename)

                    # Save the face image to the disk
                    cv2.imwrite(file_path, frame)

                    # Save just the path in MongoDB
                    try:
                        criminals_collection.insert_one({
                            'name': predicted_label,
                            'timestamp': datetime.now(),
                            'image_path': file_path  # Store the path to the image
                        })
                        print(f"Criminal detected and saved: {predicted_label}, path: {file_path}")
                    except Exception as e:
                        print(f"Failed to insert into MongoDB: {str(e)}")

                    # Send email notification
                    subject = f"Criminal Detected: {predicted_label}"
                    body = f"A criminal ({predicted_label}) has been detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. The image has been saved at: {file_path}"
                    send_email(subject, body, 'moussalasfar2000@gmail.com')  # Replace with the recipient's email

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
    # Load the image for prediction
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at path: {image_path}")
        return None, None

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the same size as training images (100x100)
    img_resized = cv2.resize(img_gray, (100, 100))
    
    # Normalize the image
    img_resized = img_resized.astype('float32') / 255.0
    
    # Reshape to match the model input shape (batch size, height, width, channels)
    img_reshaped = img_resized.reshape(-1, 100, 100, 1)

    # Predict the class
    prediction = model.predict(img_reshaped)
    
    # Get the predicted class label
    predicted_class = np.argmax(prediction)

    # Map the predicted class index to the actual label
    predicted_label = class_labels[predicted_class]

    # If the predicted class is a criminal, return "Criminal" label, else "Not Criminal"
    if predicted_label in criminal_labels:
        return predicted_label, img
    else:
        return "Not Criminal", img


def generate_frames_from_camera():
    """Générer les frames à partir de la caméra en temps réel et enregistrer les criminels dans la base de données."""
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
                    color = (0, 0, 255)  # Rouge pour les criminels

                    # Sauvegarder l'image du criminel détecté
                    result_image_path = f"static/uploads/criminal_{predicted_label}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    cv2.imwrite(result_image_path, frame)

                    # Enregistrer le criminel dans MongoDB
                    try:
                        criminals_collection.insert_one({
                            'name': predicted_label,
                            'timestamp': datetime.now(),
                            'image_path': result_image_path
                        })
                        print(f"Criminal detected and saved: {predicted_label}")
                    except Exception as e:
                        print(f"Failed to insert into MongoDB: {str(e)}")
                    
                    # Send email notification
                    subject = f"Criminal Detected: {predicted_label}"
                    body = f"A criminal ({predicted_label}) has been detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. The image has been saved at: {file_path}"
                    send_email(subject, body, 'moussalasfar2000@gmail.com')  # Replace with the recipient's email

                else:
                    label_text = "Not Criminal"
                    color = (0, 255, 0)  # Vert pour non-criminels
            else:
                label_text = "Not Criminal"
                color = (0, 255, 0)  # Vert pour inconnu

            # Dessiner le rectangle et le label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Email setup
def send_email(subject, body, to_email):
    """Send an email notification."""
    from_email = 'moussalasfar2000@gmail.com'  # Replace with your email
    from_password = 'lkzh zfmm tjfz jxzc'  # Replace with your email password or app-specific password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(from_email, from_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_camera')
def predict_camera():
    global streaming
    streaming = True  # Activer le flux pour la caméra
    return Response(generate_frames_from_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_video', methods=['POST'])
def predict_video():
    global streaming
    streaming = True  # Réinitialiser le flux pour une nouvelle vidéo

    if 'video' not in request.files:
        return "No video uploaded", 400

    video = request.files['video']
    video_path = f"uploads/{video.filename}"
    video.save(video_path)

    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        # Vérifiez si une image est incluse dans la requête
        if 'image' not in request.files:
            return "No image uploaded", 400

        # Sauvegarde de l'image envoyée
        image_file = request.files['image']
        image_path = f"uploads/{image_file.filename}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Assurez-vous que le répertoire existe
        image_file.save(image_path)

        # Processus de prédiction pour l'image
        prediction_label, processed_image = process_image(image_path)

        # Si l'image ne peut pas être traitée, retournez une erreur
        if processed_image is None:
            return f"Error processing image at path: {image_path}", 400

        # Sauvegarder l'image traitée pour l'affichage
        result_image_path = os.path.join('static', 'uploads', image_file.filename)
        os.makedirs(os.path.dirname(result_image_path), exist_ok=True)  # Assurez-vous que le répertoire existe
        cv2.imwrite(result_image_path, processed_image)

        # Liste noire de criminels 
        blacklist = ['elizabeth_holmes', 'edward_snowden']  # Remplacez par vos propres noms de criminels
        if prediction_label in blacklist:
            try:
                # Enregistrer le criminel détecté dans MongoDB
                criminals_collection.insert_one({
                    'name': prediction_label,
                    'timestamp': datetime.now(),
                    'image_path': result_image_path
                })
                print(f"Criminal detected and saved: {prediction_label}")
            except Exception as e:
                return f"Failed to insert into MongoDB: {str(e)}", 500

        # Retourner la page de résultat avec le label de prédiction
        return render_template('result.html', results=[prediction_label], image_path=result_image_path)

    except Exception as e:
        # Gérer les erreurs générales et retourner un message d'erreur
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
