import os 
import torch 
import torch.nn as nn 
from torchvision import transforms
from PIL import Image 
import timm 

# 4 sky classes as per protocol
CLASSES = ["clear","scattered","overcast","hazy"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}

def get_vision_encoder(model_name: str = "resnet18", pretrained: bool = True):
    """
    Loads an open vision encoder with frozen weiths to extrat d-dimensional features
    """

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False 

    embedding_dim = model.num_features
    return model, embedding_dim

def extract_image_features(image_path: str, model: nn.Module) -> torch.Tensor:
    """
    Preprocesses an image and gives its d-dimensional embedding vector
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze(0)

def generate_synthetic_station_data(num_stations: int = 5, samples_per_station: int = 30, d: int = 512, seed: int = 42):
    """
    Generates non-IID synthetic station data for offline development before the photo walk.
    Each station has distinct class skew (different slices of sky).
    """
    torch.manual_seed(seed)
    station_data = {}
    
    # Skew distributions: e.g. station 0 mostly clear, station 2 mostly overcast, etc.
    skew_profiles = [
        [0.7, 0.2, 0.1, 0.0],  # Station 0 (Window): mostly clear
        [0.1, 0.6, 0.2, 0.1],  # Station 1 (Alley): mostly scattered
        [0.0, 0.1, 0.8, 0.1],  # Station 2 (Park): mostly overcast
        [0.1, 0.1, 0.1, 0.7],  # Station 3 (Rooftop): mostly hazy
        [0.3, 0.3, 0.3, 0.1],  # Station 4 (Water): mixed
    ]
    
    # Fixed class prototype vectors in feature space
    class_prototypes = torch.randn(len(CLASSES), d)
    class_prototypes = class_prototypes / class_prototypes.norm(dim=-1, keepdim=True)
    
    for s in range(num_stations):
        probs = torch.tensor(skew_profiles[s])
        labels = torch.multinomial(probs, num_samples=samples_per_station, replacement=True)
        
        # Features = prototype + station bias + noise
        station_bias = torch.randn(1, d) * 0.2
        noise = torch.randn(samples_per_station, d) * 0.3
        features = class_prototypes[labels] + station_bias + noise
        features = features / features.norm(dim=-1, keepdim=True)
        
        station_data[s] = {"features": features, "labels": labels}
        
    return station_data

def extract_and_save_real_dataset(raw_dir: str = "data/raw", out_dir: str = "data/embeddings", model_name: str = "resnet18"):
    """
    Scans data/raw/station_0..4/ and data/raw/test/, extracts frozen embeddings,
    and saves .pt files directly into data/embeddings/.
    """
    os.makedirs(out_dir, exist_ok=True)
    model, d = get_vision_encoder(model_name=model_name, pretrained=True)
    
    # Process stations 0 through 4 + test set
    targets = [f"station_{i}" for i in range(5)] + ["test"]
    
    for folder_name in targets:
        folder_path = os.path.join(raw_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_name} (folder not found)")
            continue
            
        features_list, labels_list = [], []
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                cls_name = fname.split('_')[0].lower()
                if cls_name in CLASS_TO_IDX:
                    img_path = os.path.join(folder_path, fname)
                    emb = extract_image_features(img_path, model)
                    features_list.append(emb)
                    labels_list.append(CLASS_TO_IDX[cls_name])
                    
        if features_list:
            feats = torch.stack(features_list)
            labs = torch.tensor(labels_list)
            torch.save({"features": feats, "labels": labs}, os.path.join(out_dir, f"{folder_name}.pt"))
            print(f"Saved {folder_name}.pt: {feats.shape[0]} images, d={feats.shape[1]}")

    print("Embedding extraction complete!")


def load_real_station_data(embeddings_dir: str = "data/embeddings"):
    """Loads extracted real station embeddings if available."""
    station_data = {}
    for s in range(5):
        path = os.path.join(embeddings_dir, f"station_{s}.pt")
        if os.path.exists(path):
            station_data[s] = torch.load(path)
        else:
            return None  # Fall back to synthetic if real not found
    return station_data
