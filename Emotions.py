import os
from collections import deque
from time import time
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from Model import EmotionCNN

# face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

emotion_buffer = deque(maxlen=6 )
emotion_times = {}

cap = cv2.VideoCapture(0)

current_emotion = None
emotion_start_time = None

start_time = time()

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

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

        # get an "average" of emotions to prevent immediate quick burst switches
        emotion_buffer.append(emotion)

        if len(emotion_buffer) == emotion_buffer.maxlen:
            most_common_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
        else:
            most_common_emotion = emotion

        # track how long each emotion is present
        if most_common_emotion != current_emotion:
            if current_emotion is not None and emotion_start_time is not None:
                duration = time() - emotion_start_time
                if current_emotion not in emotion_times:
                    emotion_times[current_emotion] = 0
                emotion_times[current_emotion] += duration

            current_emotion = most_common_emotion
            emotion_start_time = time()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, most_common_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if current_emotion is not None and emotion_start_time is not None:
    duration = time() - emotion_start_time
    if current_emotion not in emotion_times:
        emotion_times[current_emotion] = 0
    emotion_times[current_emotion] += duration

cap.release()
cv2.destroyAllWindows()

# final plot to show comparison of how much each emotion is present

emotions = list(emotion_times.keys())
durations = list(emotion_times.values())
plt.figure(figsize=(10, 6))
plt.bar(emotions, durations, color='orange')
plt.xlabel('Emotion')
plt.ylabel('Duration (seconds)')
plt.title('Emotion Duration Tracking')
plt.xticks(rotation=45)
plt.tight_layout()
plot_path = "emotion_duration_plot.png"
plt.savefig(plot_path)
plot_img = cv2.imread(plot_path)
cv2.imshow('Emotion Duration Plot', plot_img)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()
os.remove(plot_path)
