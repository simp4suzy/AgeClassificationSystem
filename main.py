import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
MODEL_PATH = "models/age_classification_final_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

class AgeClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("900x700")
        self.root.config(bg="#2E2E2E")
        self.show_splash()

    def show_splash(self):
        """Display an animated splash screen."""
        for widget in self.root.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.root, width=900, height=700)
        canvas.pack(fill="both", expand=True)

        gradient_img = self.create_gradient((900, 700), "#282828", "#4CAF50")
        gradient_bg = ImageTk.PhotoImage(gradient_img)
        canvas.create_image(0, 0, anchor="nw", image=gradient_bg)
        self.root.gradient_bg = gradient_bg

        self.splash_text = tk.Label(self.root, text="", font=("Arial", 48, "bold"), fg="white", bg="#4CAF50")
        self.splash_text.place(relx=0.5, rely=0.5, anchor="center")
        self.animate_text("Welcome to AgeReveal", 0)

    def animate_text(self, text, index):
        if index < len(text):
            self.splash_text.config(text=text[:index + 1])
            self.root.after(100, self.animate_text, text, index + 1)
        else:
            self.root.after(2000, self.init_main_ui)

    def create_gradient(self, size, color1, color2):
        base = Image.new("RGB", size, color1)
        top = Image.new("RGB", size, color2)
        mask = Image.new("L", size)
        mask_data = [int(255 * (y / size[1])) for y in range(size[1]) for _ in range(size[0])]
        mask.putdata(mask_data)
        base.paste(top, (0, 0), mask)
        return base

    def init_main_ui(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.attributes("-fullscreen", False)
        self.root.geometry("900x700")
        self.root.title("AgeReveal - Age Classification System")

        self.title_label = tk.Label(self.root, text="AgeReveal - Age Classification System", font=("Arial", 20, "bold"),
                                    bg="#1F1F1F", fg="white", pady=10)
        self.title_label.pack(fill="x")

        self.video_frame = tk.Frame(self.root, bg="#333333", relief="ridge", bd=5)
        self.video_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.video_label = tk.Label(self.video_frame, bg="#333333")
        self.video_label.pack(fill="both", expand=True)

        self.control_frame = tk.Frame(self.root, bg="#1F1F1F")
        self.control_frame.pack(pady=20)

        self.start_button = tk.Button(self.control_frame, text="Start Detection", command=self.start_camera,
                                       width=15, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                       relief="groove", bd=3)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.snapshot_button = tk.Button(self.control_frame, text="Save Snapshot", command=self.save_snapshot,
                                         width=15, height=2, bg="#2196F3", fg="white", font=("Arial", 12, "bold"),
                                         relief="groove", bd=3)
        self.snapshot_button.grid(row=0, column=1, padx=10, pady=10)
        self.snapshot_button.grid_remove()

        self.help_button = tk.Button(self.control_frame, text="Help", command=self.show_help,
                                     width=15, height=2, bg="#FF9800", fg="white", font=("Arial", 12, "bold"),
                                     relief="groove", bd=3)
        self.help_button.grid(row=0, column=2, padx=10, pady=10)

        self.exit_button = tk.Button(self.control_frame, text="Exit", command=self.quit_app,
                                     width=15, height=2, bg="#F44336", fg="white", font=("Arial", 12, "bold"),
                                     relief="groove", bd=3)
        self.exit_button.grid(row=0, column=3, padx=10, pady=10)

        self.status_label = tk.Label(self.root, text="Status: Ready", bg="#2E2E2E", fg="white", font=("Arial", 12), anchor="w")
        self.status_label.place(relx=0.01, rely=0.86, anchor="sw")

        self.cap = None
        self.snapshot_frame = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera")
            return

        self.start_button.grid_remove()
        self.snapshot_button.grid()
        self.update_status("Starting camera...")
        self.update_frame()

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=8,
                    minSize=(60, 60),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                if len(faces) > 0:
                    self.update_status("Detecting face/s...")
                    for (x, y, w, h) in faces:
                        face = frame[y:y + h, x:x + w]
                        predicted_age = self.predict_age(face)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, predicted_age, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    self.update_status("Failing to recognize face/s...")

                self.snapshot_frame = frame

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def save_snapshot(self):
        if self.snapshot_frame is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, self.snapshot_frame)
                self.update_status(f"Snapshot saved to {file_path}")

    def predict_age(self, face_img):
        face_img = cv2.resize(face_img, (64, 64))
        face_img = np.expand_dims(face_img / 255.0, axis=0)
        predictions = model.predict(face_img)
        predicted_class = np.argmax(predictions)
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
        return f"Age: {bins[predicted_class]}-{bins[predicted_class + 1]}"

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

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AgeClassificationApp(root)
    root.mainloop()