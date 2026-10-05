from dataset import FaceSpoofDataset

# TODO: Put your actual raw dataset folder path here (e.g., r"C:\Users\minli\Downloads\dataset")
DATASET_ROOT = r"C:\Users\minli\Downloads\dataset"
TRAIN_JSON = r"C:\Users\minli\Downloads\dataset\train_split.json"

# Initialize our dataset
train_dataset = FaceSpoofDataset(json_path=TRAIN_JSON, root_dir=DATASET_ROOT)

print(f"Total training images found: {len(train_dataset)}")

# Test pulling the very first image and label
sample_image, sample_label = train_dataset[0]
print(f"Successfully loaded sample 0!")
print(f"Image format/size: {sample_image.size}")
print(f"Image label: {sample_label} ('1' = Live, '0' = Spoof)")