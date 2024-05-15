# Download Tiny ImageNet data
import requests
import io
import zipfile

print("Download image data...")
response = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
os.makedirs("./data", exist_ok=True)
zip_file.extractall("./data")

# Move image data
import os
import shutil

print("Move image data...")
# move train images
os.makedirs("./data/train", exist_ok=True)

for label in os.listdir("./data/tiny-imagenet-200/train"):
    image_files = os.listdir(f"./data/tiny-imagenet-200/train/{label}/images")
    os.makedirs(f"./data/train/{label}", exist_ok=True)
    for file in image_files:
        shutil.move(
            f"./data/tiny-imagenet-200/train/{label}/images/{file}",
            f"./data/train/{label}/{file}"
        )

# # move test images
os.makedirs("./data/test", exist_ok=True)

for file in os.listdir("./data/tiny-imagenet-200/test/images"):
    shutil.move(
        f"./data/tiny-imagenet-200/test/images/{file}",
        f"./data/test/{file}",
    )

shutil.rmtree("./data/tiny-imagenet-200")