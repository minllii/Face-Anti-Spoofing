import cv2
import torch
import torch.nn.functional as F
from model import SimpleCNN
from torchvision import transforms

# --- CONFIGURATION ---
MODEL_PATH = "spoof_cnn_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading trained model on {DEVICE}...")
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# --- DATA TRANSFORM FOR WEBCAM ---
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# Load our local Haar Cascade file directly from the project folder
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
  print("Error: Could not load Haar Cascade XML file.")
  exit()

# Open webcam
cap = cv2.VideoCapture(0)
print("\nStarting webcam... Press 'q' on your keyboard to quit.")

while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame from webcam. Exiting...")
    break

  # Convert frame to grayscale for face detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
  )

  for x, y, w, h in faces:
    # Crop the face region from the color frame
    face_crop = frame[y : y + h, x : x + w]

    if face_crop.size == 0:
      continue

    # Preprocess the face crop for our CNN model
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    input_tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

    # Run inference
    with torch.no_grad():
      outputs = model(input_tensor)
      probabilities = F.softmax(outputs, dim=1)
      confidence, predicted_class = torch.max(probabilities, dim=1)

      label_idx = predicted_class.item()
      conf_score = confidence.item() * 100

    # --- CONFIDENCE THRESHOLD TUNING ---
    # If the model is less than 75% sure, let's treat it as "Uncertain" to stop flickering
    CONFIDENCE_THRESHOLD = 75.0

    if conf_score < CONFIDENCE_THRESHOLD:
      label_text = f"UNCERTAIN ({conf_score:.1f}%)"
      box_color = (255, 165, 0)  # Orange for Uncertain
    else:
      # Interpret label: 1 = Live, 0 = Spoof
      if label_idx == 1:
        label_text = f"LIVE ({conf_score:.1f}%)"
        box_color = (0, 255, 0)  # Green for Live
      else:
        label_text = f"SPOOF ({conf_score:.1f}%)"
        box_color = (0, 0, 255)  # Red for Spoof

    # Interpret label: 1 = Live, 0 = Spoof
    if label_idx == 1:
      label_text = f"LIVE ({conf_score:.1f}%)"
      box_color = (0, 255, 0)  # Green for Live
    else:
      label_text = f"SPOOF ({conf_score:.1f}%)"
      box_color = (0, 0, 255)  # Red for Spoof

    # Draw bounding box and label
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    cv2.putText(
        frame,
        label_text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        box_color,
        2,
    )

  # Display the live window
  cv2.imshow("Face Anti-Spoofing Real-Time Test", frame)

  # Quit when 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")