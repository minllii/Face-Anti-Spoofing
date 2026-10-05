import os
import cv2

# Create folders to store your custom images
os.makedirs("custom_data/live", exist_ok=True)
os.makedirs("custom_data/spoof", exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

print("--- DATA COLLECTION UTILITY ---")
print("Instructions:")
print("  - Position your actual face and press 'l' to save as LIVE.")
print("  - Hold your phone up and press 's' to save as SPOOF.")
print("  - Press 'q' to quit when you have enough samples (aim for ~30-50 of each).")

live_count = len(os.listdir("custom_data/live"))
spoof_count = len(os.listdir("custom_data/spoof"))

while True:
  ret, frame = cap.read()
  if not ret:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
  )

  display_frame = frame.copy()

  for x, y, w, h in faces:
    # Draw a blue tracking box while collecting
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face_crop = frame[y : y + h, x : x + w]

    # Instructions on screen
    cv2.putText(
        display_frame,
        f"Live Saved: {live_count} | Spoof Saved: {spoof_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display_frame,
        "Press 'l' (Live) or 's' (Spoof)",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

  cv2.imshow("Data Collector", display_frame)

  key = cv2.waitKey(1) & 0xFF

  if key == ord("q"):
    break

  elif key == ord("l") and len(faces) > 0:
    # Save live face crop
    x, y, w, h = faces[0]
    face_crop = frame[y : y + h, x : x + w]
    img_path = os.path.join("custom_data", "live", f"live_{live_count}.jpg")
    cv2.imwrite(img_path, face_crop)
    live_count += 1
    print(f"[SAVED LIVE] Total live: {live_count}")

  elif key == ord("s") and len(faces) > 0:
    # Save spoof face crop
    x, y, w, h = faces[0]
    face_crop = frame[y : y + h, x : x + w]
    img_path = os.path.join("custom_data", "spoof", f"spoof_{spoof_count}.jpg")
    cv2.imwrite(img_path, face_crop)
    spoof_count += 1
    print(f"[SAVED SPOOF] Total spoof: {spoof_count}")

cap.release()
cv2.destroyAllWindows()
print("Collection finished!")