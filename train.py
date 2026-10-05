import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import FaceSpoofDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision import transforms

# --- CONFIGURATION & HYPERPARAMETERS ---
# TODO: Put your actual dataset folder path here
DATASET_ROOT = r"C:\Users\minli\Downloads\dataset"
TRAIN_JSON = os.path.join(DATASET_ROOT, "train_split.json")
VAL_JSON = os.path.join(DATASET_ROOT, "val_split.json")

BATCH_SIZE = 32  # Number of images processed together in one step
EPOCHS = 5  # Number of full passes over the dataset (start small for testing)
LEARNING_RATE = 0.001  # How fast the model updates its weights

# --- DATA TRANSFORMS WITH AUGMENTATION ---
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        # Add random lighting and color changes so the model isn't fooled by brightness!
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)


def main():
  print("Initializing dataset and dataloaders...")

  train_dataset = FaceSpoofDataset(
    json_path=TRAIN_JSON,
    root_dir=DATASET_ROOT,
    transform=transform,
    is_train=True,  # This tells it to load your 110 custom images too!
  )
  val_dataset = FaceSpoofDataset(
      json_path=VAL_JSON, root_dir=DATASET_ROOT, transform=transform
  )

  # DataLoader handles batching and shuffling the data automatically
  train_loader = DataLoader(
      train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
  )
  val_loader = DataLoader(
      val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
  )

  print(f"Training batches: {len(train_loader)}")
  print(f"Validation batches: {len(val_loader)}")

  # Set up device (use GPU if available for faster training, otherwise CPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Initialize Model, Loss Function, and Optimizer
  model = SimpleCNN().to(device)
  criterion = nn.CrossEntropyLoss()  # Standard loss for classification
  optimizer = optim.Adam(
      model.parameters(), lr=LEARNING_RATE
  )  # Optimizer that updates weights

  # --- TRAINING LOOP ---
  for epoch in range(EPOCHS):
    model.train()  # Put model in training mode (enables Dropout)
    running_loss = 0.0

    print(f"\n--- Epoch [{epoch+1}/{EPOCHS}] Starting ---")

    for batch_idx, (images, labels) in enumerate(train_loader):
      # Move data to GPU or CPU
      images, labels = images.to(device), labels.to(device)

      # Step 1: Clear old gradients
      optimizer.zero_grad()

      # Step 2: Forward pass (get model predictions)
      outputs = model(images)

      # Step 3: Calculate loss (how wrong was the prediction?)
      loss = criterion(outputs, labels)

      # Step 4: Backward pass (calculate adjustments)
      loss.backward()

      # Step 5: Update model weights
      optimizer.step()

      running_loss += loss.item()

      # Print progress every 500 batches so we know it's working
      if (batch_idx + 1) % 500 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
        )

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Complete. Average Loss: {avg_train_loss:.4f}")

  # Save the trained model weights
  torch.save(model.state_dict(), "spoof_cnn_model.pth")
  print("\nTraining complete! Model saved successfully as 'spoof_cnn_model.pth'.")


if __name__ == "__main__":
  main()