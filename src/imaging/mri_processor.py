import os
from PIL import Image

def load_mri_sample():

    root = "data/imaging/mri"

    if not os.path.exists(root):
        return None

    for path, dirs, files in os.walk(root):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                img_path = os.path.join(
                    path,
                    file
                )

                try:
                    img = Image.open(img_path)
                    return img

                except:
                    pass

    return None