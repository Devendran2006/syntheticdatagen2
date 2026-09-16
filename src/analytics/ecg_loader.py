import os
import wfdb


def load_ecg_sample():

    base_path = r"data/signals/ecg"

    for root, dirs, files in os.walk(base_path):

        for file in files:

            if file.endswith(".hea"):

                record = os.path.join(
                    root,
                    file.replace(".hea", "")
                )

                signal, meta = wfdb.rdsamp(record)

                return signal, meta

    return None, None