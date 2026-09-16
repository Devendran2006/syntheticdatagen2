import subprocess
import time

steps = [

    ("PII Detection",
     "python src/pii_detection.py"),

    ("Preprocessing",
     "python src/preprocessing.py"),

    ("CTGAN Training",
     "python src/ctgan_train.py"),

     ("Model Registry",
      "python src/model_registry.py"),

    ("Evaluation",
     "python src/evaluation.py"),

    ("Privacy Score",
     "python src/privacy_score.py"),

    ("Correlation Similarity",
     "python src/correlation_similarity.py"),

    ("Quality Report",
     "python src/quality_report.py"),

    (
        "Save Metrics",
        "python src/save_metrics.py"
        )
]

for name, cmd in steps:

    print("\n" + "="*50)
    print(f"Running: {name}")
    print("="*50)

    result = subprocess.run(
        cmd,
        shell=True
    )

    if result.returncode != 0:

        print(f"Failed at {name}")
        break

print("\nPipeline Completed")