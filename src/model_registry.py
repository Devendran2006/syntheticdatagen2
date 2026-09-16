import os
import shutil
from datetime import datetime

os.makedirs(
    "artifacts",
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

source_model = "models/ctgan_model.pkl"

destination_model = (
    f"artifacts/model_{timestamp}.pkl"
)

shutil.copy(
    source_model,
    destination_model
)

print(
    f"Model Saved: {destination_model}"
)