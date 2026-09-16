import subprocess
import time

steps = [

    ("PII Detection",
     "python src/pii_detection.py"),

    ("Preprocessing",
     "python src/preprocessing.py"),

    ("CTGAN Training",
     "python src/ctgan_train.py"),

    ("Evaluation",
     "python src/evaluation.py"),

    ("ML Validation Real",
     "python src/ml_validation_real.py"),

    ("ML Validation Synthetic",
     "python src/ml_synthetic.py"),

    ("Privacy Score",
     "python src/privacy_score.py"),

    ("Correlation Similarity",
     "python src/correlation_similarity.py"),

    ("Quality Report",
     "python src/quality_report.py")

]

start_time = time.time()

for step_no, (name, cmd) in enumerate(steps, start=1):

    print("\n" + "=" * 60)
    print(f"STEP {step_no}/{len(steps)} : {name}")
    print("=" * 60)

    result = subprocess.run(
        cmd,
        shell=True
    )

    if result.returncode != 0:

        print(f"\nFAILED AT: {name}")
        break

else:

    print("\nALL PIPELINE STEPS COMPLETED SUCCESSFULLY")

end_time = time.time()

print("\nTotal Execution Time:",
      round((end_time - start_time) / 60, 2),
      "minutes")