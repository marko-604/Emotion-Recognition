import os
import cv2
import torch
import requests
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import deque
from time import time
from PIL import Image
from torchvision.models import resnet50
import torch.nn as nn


# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load emotion model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=False)  # Ensure same architecture
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 7)  # Match trained model
)
model.load_state_dict(torch.load("resnet50_emotion_model.pth", map_location=device))
model.to(device)
model.eval()


# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_label_map = {label: idx for idx, label in enumerate(emotion_labels)}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure consistent size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Use same normalization as training
])


# Ask user for input mode
mode = input("Enter 'w' for webcam or 'v' for video file: ").strip().lower()

if mode == 'v':
    video_path = input("Enter video file name (with extension): ").strip()
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

# Ask player what they intend to do
player_intent = input("What are you about to do? ")

# Choose LLM method
llm_choice = input("Use (1) Ollama (local): ").strip()

# Emotion tracking
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

# Function to query LLM
def query_llm(intent, emotions):
    if llm_choice == "1":  # Ollama (Local)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "mistral",
            "prompt": f"Player intent: '{intent}'. Here is their emotion data during the desired event:\n{emotions}\n\nAnalyze how their emotions relate to intent.",
            "stream": False
        }
    elif llm_choice == "2":  # OpenAI GPT
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "system", "content": "You are an AI analyzing player emotions."},
                         {"role": "user", "content": f"Player intent: '{intent}'. Emotion data:\n{emotions}\nAnalyze it."}],
            "max_tokens": 200
        }
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    else:
        return "Invalid LLM choice."

    try:
        response = requests.post(url, json=payload, headers=headers if llm_choice == "2" else {})
        return response.json()["response"] if llm_choice == "1" else response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

# Process emotion data
emotion_summary = {k: round(v / 60, 2) for k, v in emotion_times.items()}
llm_response = query_llm(player_intent, emotion_summary)

# Show LLM response
print("\n--- LLM Analysis ---")
print(llm_response)

# Plot emotion, overall durations
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
