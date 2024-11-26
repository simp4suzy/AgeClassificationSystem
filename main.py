import os
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the trained model
MODEL_PATH = "models/age_classification_final_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Initialize the model globally
model = load_model(MODEL_PATH)

# Age bins for prediction
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]

def predict_age(face_img):
    """
    Predict the age group of a face image using the model.
    :param face_img: Cropped face image
    :return: Predicted age range (e.g., "20-30")
    """
    face_img = cv2.resize(face_img, (64, 64))
    face_img = np.expand_dims(face_img / 255.0, axis=0)  # Normalize and add batch dimension
    predictions = model.predict(face_img)
    predicted_class = np.argmax(predictions)
    return f"{bins[predicted_class]}-{bins[predicted_class + 1]}"

class AgeClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Age Classification System")
        self.root.geometry("900x700")
        self.root.config(bg="#2E2E2E")

        # Frames for layout
        self.video_frame = tk.Frame(self.root, bg="#2E2E2E")
        self.video_frame.pack(pady=20)

        self.control_frame = tk.Frame(self.root, bg="#2E2E2E")
        self.control_frame.pack()

        # Video display
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Buttons
        self.start_button = tk.Button(self.control_frame, text="Start Detection", command=self.start_camera,
                                       width=15, height=2, bg="#4CAF50", fg="white", font=("Arial", 12),
                                       relief="raised", bd=3)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.snapshot_button = tk.Button(self.control_frame, text="Save Snapshot", command=self.save_snapshot,
                                          width=15, height=2, bg="#2196F3", fg="white", font=("Arial", 12),
                                          relief="raised", bd=3, state="disabled")
        self.snapshot_button.grid(row=0, column=1, padx=10, pady=10)

        self.help_button = tk.Button(self.control_frame, text="Help", command=self.show_help,
                                     width=15, height=2, bg="#FF9800", fg="white", font=("Arial", 12),
                                     relief="raised", bd=3)
        self.help_button.grid(row=0, column=2, padx=10, pady=10)

        self.exit_button = tk.Button(self.control_frame, text="Exit", command=self.quit_app,
                                     width=15, height=2, bg="#F44336", fg="white", font=("Arial", 12),
                                     relief="raised", bd=3)
        self.exit_button.grid(row=0, column=3, padx=10, pady=10)

        # Status bar
        self.status_label = tk.Label(self.root, text="Status: Ready", bg="#2E2E2E", fg="white", font=("Arial", 12),
                                     anchor="w")
        self.status_label.pack(fill="x", pady=10)

        self.cap = None  # Video capture object
        self.snapshot_frame = None

    def start_camera(self):
        # Start the webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera")
            return
        self.snapshot_button.config(state="normal")
        self.update_frame()
        self.update_status("Detecting faces...")

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to grayscale for more reliable face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Equalize histogram to handle varying lighting conditions
                gray = cv2.equalizeHist(gray)

                # Load the Haar Cascade for face detection
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

                # Detect faces with stricter parameters to minimize false positives
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,  # Adjust for better accuracy
                    minNeighbors=8,  # Higher value reduces false detections
                    minSize=(60, 60),  # Ignore small regions unlikely to be faces
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (x, y, w, h) in faces:
                    # Crop the detected face
                    face = frame[y:y + h, x:x + w]

                    # Optional: Additional filtering to ensure human face detection
                    if w / h < 0.8 or w / h > 1.2:  # Skip non-square faces
                        continue

                    # Predict the age range for the face
                    predicted_age = predict_age(face)

                    # Draw a rectangle and annotate with the predicted age range
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_age, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Save the frame for potential snapshots
                self.snapshot_frame = frame

                # Convert frame to RGB for display in Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        # Schedule the next frame update
        self.root.after(10, self.update_frame)

    def save_snapshot(self):
        if self.snapshot_frame is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, self.snapshot_frame)
                self.update_status(f"Snapshot saved to {file_path}")

    def show_help(self):
        messagebox.showinfo("Help", "1. Click 'Start Detection' to begin.\n"
                                    "2. Faces with age predictions will appear on the video feed.\n"
                                    "3. Click 'Save Snapshot' to save a frame.\n"
                                    "4. Click 'Exit' to close the application.")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def quit_app(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.quit()

    def __del__(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AgeClassificationApp(root)
    root.mainloop()