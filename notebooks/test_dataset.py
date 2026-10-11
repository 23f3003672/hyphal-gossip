import sys
sys.path.append(".")

from src.dataset import generate_synthetic_station_data, CLASSES

data = generate_synthetic_station_data(num_stations=5,samples_per_station=30, d=512)
print("Generated station data successfully")

for station_id,d in data.items():
    feats = d["features"]
    labels = d["labels"]
    counts = {CLASSES[i]: (labels == i).sum().item() for i in range(len(CLASSES))}
    print(f" Station {station_id}: shape={feats.shape}, class distribution = {counts}")