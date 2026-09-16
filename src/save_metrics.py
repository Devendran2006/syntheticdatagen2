import pandas as pd
from datetime import datetime
import os

quality = (
    open(
        "outputs/quality_score.txt"
    )
    .read()
    .strip()
    .replace("Quality Score:", "")
    .replace("%", "")
    .strip()
)

privacy = (
    open(
        "outputs/privacy_score.txt"
    )
    .read()
    .strip()
    .replace("Privacy Score:", "")
    .replace("%", "")
    .strip()
)

correlation = (
    open(
        "outputs/correlation_score.txt"
    )
    .read()
    .strip()
    .replace("Correlation Similarity:", "")
    .replace("%", "")
    .strip()
)
new_row = pd.DataFrame({
    "Timestamp": [datetime.now()],
    "Quality Score": [float(quality)],
    "Privacy Score": [float(privacy)],
    "Correlation Similarity": [float(correlation)]
})

file_path = "metrics/training_history.csv"

if os.path.exists(file_path):

    old_data = pd.read_csv(file_path)

    history = pd.concat(
        [old_data, new_row],
        ignore_index=True
    )

else:

    history = new_row

history.to_csv(
    file_path,
    index=False
)

history.tail(1).to_csv(
    "metrics/latest_metrics.csv",
    index=False
)

print("Training History Updated")