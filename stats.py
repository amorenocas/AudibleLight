import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PATH = Path("spatial_scenes_dcase_synthetic/metadata_dev")
SPLIT = "dev-test-alight"

azimuths = []
elevations = []
distances = []
for filename in os.listdir(PATH / SPLIT):
    if "_doa" in filename:
        with open(PATH / SPLIT / filename, "r") as f:
            doa_dict = json.load(f)
        azimuths.append(doa_dict["azimuth"])
        elevations.append(doa_dict["elevation"])
        distances.append(doa_dict["distance"])
plt.figure(figsize=(8, 4))
sns.histplot(azimuths, bins=50, kde=True)
plt.title(f"Distribution of Azimuths ({SPLIT})")
plt.xlabel("Azimuth angle (degrees)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 4))
sns.histplot(elevations, bins=50, kde=True)
plt.title(f"Distribution of Elevations ({SPLIT})")
plt.xlabel("Elevation angle (degrees)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 4))
sns.histplot(distances, bins=50, kde=True)
plt.title(f"Distribution of Distances ({SPLIT})")
plt.xlabel("Distance (meters)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
