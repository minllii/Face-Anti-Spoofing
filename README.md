# Real-Time Face Anti-Spoofing System

A lightweight, customized Convolutional Neural Network (CNN) built with PyTorch and OpenCV to detect and prevent presentation attacks (face spoofing via phone/screen replays) in real time.

---

## 🚀 Features
* **Real-Time Inference:** Fast webcam processing using OpenCV and PyTorch.
* **Custom Environment Adaptation:** Fine-tuned with local user samples to eliminate domain shift caused by room lighting, distance variations, and glasses reflections.
* **Anti-Spoof Classification:** Distinguishes live human faces from digital screen replay attacks (e.g., smartphones).
* **Robust Augmentation:** Trained using `ColorJitter` and `RandomHorizontalFlip` to resist lighting variations.

---

## 🛠️ Tech Stack
* **Python**
* **PyTorch & torchvision** (Deep Learning CNN model)
* **OpenCV** (Webcam capture & face detection via Haar Cascades)
* **Pillow (PIL)** (Image processing)

---

## 📂 Project Structure
```text
Face-Anti-Spoofing/
│
├── dataset.py            # Custom dataset handler (supports base dataset & personal samples)
├── model.py              # CNN Architecture (SimpleCNN)
├── train.py              # Training loop with augmentations
├── inference.py          # Real-time webcam evaluation script
├── collect_data.py       # Personal data capture utility for fine-tuning
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation