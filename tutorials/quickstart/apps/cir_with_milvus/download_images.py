from datasets import load_dataset
from cfg import Config
import os


def download_images():
    config = Config()
    with open("categories.txt") as fw:
        lines = fw.readlines()
        for line in lines:
            l = line.strip()
            meta_dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{l}", split="full"
            )
            for i in range(config.imgs_per_category):
                if len(meta_dataset[i]["images"]["large"]) > 0:
                    img_name = meta_dataset[i]["images"]["large"][0]
                    basename = os.path.basename(img_name)
                    if os.path.exists(f"{config.download_path}/{basename}") is False:
                        os.system(
                            f"wget {img_name} -P {config.download_path} --no-check-certificate"
                        )


if __name__ == "__main__":
    download_images()
