import os
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import deque
from time import time
from PIL import Image
from Model import EmotionCNN

# Face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("../resnet50_emotion_model.pth", map_location=device))
model.eval()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_label_map = {label: idx for idx, label in enumerate(emotion_labels)}

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Ask user for input mode
mode = input("Enter 'w' for webcam or 'v' for video file: ").strip().lower()

if mode == 'v':
    video_path = input("Enter video file name (with extension): ").strip()
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

emotion_buffer = deque(maxlen=6)
emotion_times = {}
time_emotions = []
time_stamps = []

current_emotion = None
emotion_start_time = None
start_time = time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) if mode == 'w' else frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

        emotion_buffer.append(emotion)
        most_common_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

        if most_common_emotion != current_emotion:
            if current_emotion is not None and emotion_start_time is not None:
                duration = time() - emotion_start_time
                emotion_times[current_emotion] = emotion_times.get(current_emotion, 0) + duration

            current_emotion = most_common_emotion
            emotion_start_time = time()

        time_stamps.append(time() - start_time)
        time_emotions.append(emotion_label_map[most_common_emotion])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, most_common_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if current_emotion is not None and emotion_start_time is not None:
    duration = time() - emotion_start_time
    emotion_times[current_emotion] = emotion_times.get(current_emotion, 0) + duration

cap.release()
cv2.destroyAllWindows()

# Plot emotion duration
plt.figure(figsize=(10, 6))
plt.bar(emotion_times.keys(), [v / 60 for v in emotion_times.values()], color='orange')
plt.xlabel('Emotion')
plt.ylabel('Duration (minutes)')
plt.title('Emotion Duration Tracking')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("emotion_duration.png")
cv2.imshow('Emotion Duration', cv2.imread("emotion_duration.png"))
cv2.waitKey(0)
os.remove("emotion_duration.png")
cv2.destroyAllWindows()

# Plot emotion over time
plt.figure(figsize=(10, 6))
plt.plot([ts / 60 for ts in time_stamps], time_emotions, color='blue')
plt.xlabel('Time (minutes)')
plt.ylabel('Emotion')
plt.title('Emotion Over Time')
plt.yticks(range(7), emotion_labels)
plt.tight_layout()
plt.savefig("emotion_time.png")
cv2.imshow('Emotion Over Time', cv2.imread("emotion_time.png"))
cv2.waitKey(0)
os.remove("emotion_time.png")
cv2.destroyAllWindows()
