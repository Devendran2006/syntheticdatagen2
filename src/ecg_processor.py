import wfdb
import os

def load_ecg_sample():

    root = "data/signals/ecg"

    for path, dirs, files in os.walk(root):

        for file in files:

            if file.endswith(".hea"):

                record = os.path.join(path, file[:-4])

                signal, meta = wfdb.rdsamp(record)

                return signal, meta

    return None, None