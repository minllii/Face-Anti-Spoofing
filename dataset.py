import json
import os
from PIL import Image
from torch.utils.data import Dataset


class FaceSpoofDataset(Dataset):

  def __init__(self, json_path, root_dir, transform=None, is_train=False):
    """Args:

        json_path (string): Path to the train_split.json or val_split.json file.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transforms to apply.
        is_train (bool): If True, also load custom_data folder for training.
    """
    with open(json_path, "r") as f:
      self.data = json.load(f)

    self.root_dir = root_dir
    self.transform = transform

    # If this is the training dataset, inject our custom personal images!
    if is_train:
      custom_live_dir = os.path.join("custom_data", "live")
      custom_spoof_dir = os.path.join("custom_data", "spoof")

      custom_items = []

      if os.path.exists(custom_live_dir):
        for img_name in os.listdir(custom_live_dir):
          if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            custom_items.append({
                "path": os.path.join(custom_live_dir, img_name),
                "label": 1,  # 1 for Live
                "is_absolute": True,  # Flag to handle absolute path
            })

      if os.path.exists(custom_spoof_dir):
        for img_name in os.listdir(custom_spoof_dir):
          if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            custom_items.append({
                "path": os.path.join(custom_spoof_dir, img_name),
                "label": 0,  # 0 for Spoof
                "is_absolute": True,
            })

      # OVERSAMPLE custom images so the model pays closer attention to them (repeat 5x)
      oversampled_custom_items = custom_items * 5

      # Append custom images to the dataset list
      self.data.extend(oversampled_custom_items)
      print(
          f"Loaded {len(custom_items)} unique custom images (Oversampled to"
          f" {len(oversampled_custom_items)} entries)!"
      )

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]

    # Check if it's a custom image or a main dataset image
    if item.get("is_absolute", False):
      img_path = item["path"]
    else:
      rel_path = item["path"].replace("dataset\\", "").replace("dataset/", "")
      img_path = os.path.join(self.root_dir, rel_path)

    # Load image using PIL
    image = Image.open(img_path).convert("RGB")
    label = item["label"]

    if self.transform:
      image = self.transform(image)

    return image, label