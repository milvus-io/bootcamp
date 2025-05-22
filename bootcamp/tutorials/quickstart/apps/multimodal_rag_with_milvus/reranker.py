import json
import os
import cv2
import numpy as np
import base64
import requests
from openai import AzureOpenAI


class GenerativeReranker:
    """
    a class to find the best product for a person by creating a panoramic view image of 25 products,
    arranging them into a 5x5 grid, and utilizing an LLM to rank these products
    """

    def __init__(
        self,
        rowCount: int = 5,
        dim: tuple = (300, 300),
        cache_file: str = "cache.json",
        api_type: str = "openai",
    ) -> None:
        """
        initializes the GenerativeReranker with the specified row count, dimension, and a path to cache file

        args:
            rowCount: Number of images per row
            dim: (width, height) of each thumbnail image
            cache_file: file path to store cached responses
            api_type: type of API to use ('openai' or 'azure_openai')
        """
        self.rowCount = rowCount
        self.dim = dim
        self.combined_image = None
        self.api_type = api_type
        self.cache_file = cache_file
        self.cache = self.load_cache()

        # set API key based on the specified API type
        if self.api_type == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api_type == "azure_openai":
            self.api_key = os.getenv("AZURE_OPENAI_KEY")
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

            api_version = "2023-12-01-preview"  # this might change in the future
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=api_version,
                base_url=f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment_name}",
            )
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

    def __call__(self, query, caption, images, infos) -> list[int]:
        """
        combines the query image and 24 retrieved images into a 5x5 panoramic view
        inputs the combined image, caption, and infos to LLM
        reranks the images based on these inputs, excluding the query image

        args:
            query: the query image
            caption: description of the desired output images
            images: list of 24 images retrieved from the dataset
            infos: list of related information about each image and the person (e.g. hobbies, passion)

        returns:
            list of indices of the newly ranked images from most matched to least matched, excluding the query image
        """
        cache_key = self.generate_cache_key(caption, infos)
        if cache_key in self.cache:
            ranked_indices, _ = self.cache[cache_key]
        else:
            self.combined_image = self.create_panoramic_view([query] + images)
            cv2.imwrite("combined_image.jpg", self.combined_image)
            if self.api_type == "openai":
                ranked_indices, explanation = self.generate_ranking_explanation(
                    caption, infos
                )
            elif self.api_type == "azure_openai":
                ranked_indices, explanation = (
                    self.generate_ranking_explanation_azure_openai(caption, infos)
                )
            else:
                pass

            self.cache[cache_key] = (ranked_indices, explanation)
            self.save_cache()
        return [idx for idx in ranked_indices]

    def get_best_item(self, query, caption, images, infos) -> int:
        """
        returns:
            index of the most matched image
        """
        ranked_indices = self.__call__(query, caption, images, infos)
        return ranked_indices[0]

    def explain(self, query, caption, images, infos) -> str:
        """
        provides an explanation of why the best item is chosen based on the query, caption, images, and infos

        returns:
            explanation of why the best item is the best choice
        """
        cache_key = self.generate_cache_key(caption, infos)
        if cache_key in self.cache:
            _, explanation = self.cache[cache_key]
        else:
            if self.combined_image is None:
                self.combined_image = self.create_panoramic_view([query] + images)
            ranked_indices, explanation = self.generate_ranking_explanation(
                caption, infos
            )
            self.cache[cache_key] = (ranked_indices, explanation)
            self.save_cache()

        return explanation

    def create_panoramic_view(self, images: list) -> np.ndarray:
        """
        creates a 5x5 panoramic view image from a list of images

        args:
            images: list of images to be combined

        returns:
            np.ndarray: the panoramic view image
        """
        img_width, img_height = self.dim
        panoramic_width = img_width * self.rowCount
        panoramic_height = img_height * self.rowCount
        panoramic_image = np.full(
            (panoramic_height, panoramic_width, 3), 255, dtype=np.uint8
        )

        # create and resize the query image with a blue border
        query_image = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)
        resized_image = cv2.resize(images[0], (img_width, img_height))

        border_size = 10
        blue = (255, 0, 0)  # blue color in BGR
        bordered_query_image = cv2.copyMakeBorder(
            resized_image,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=blue,
        )

        query_image[img_height * 2 : img_height * 3, 0:img_width] = cv2.resize(
            bordered_query_image, (img_width, img_height)
        )

        # add text "query" below the query image
        text = "query"
        font_scale = 1
        font_thickness = 2
        text_org = (10, img_height * 3 + 30)
        cv2.putText(
            query_image,
            text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            blue,
            font_thickness,
            cv2.LINE_AA,
        )

        # combine the rest of the images into the panoramic view
        for i, image in enumerate(images[1:]):
            image = cv2.resize(image, (img_width - 4, img_height - 4))
            row = i // self.rowCount
            col = i % self.rowCount
            start_row = row * img_height
            start_col = col * img_width

            border_size = 2
            bordered_image = cv2.copyMakeBorder(
                image,
                border_size,
                border_size,
                border_size,
                border_size,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            panoramic_image[
                start_row : start_row + img_height, start_col : start_col + img_width
            ] = bordered_image

            # add red index numbers to each image
            text = str(i)
            org = (start_col + 50, start_row + 30)
            (font_width, font_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )

            top_left = (org[0] - 48, start_row + 2)
            bottom_right = (org[0] - 48 + font_width + 5, org[1] + baseline + 5)

            cv2.rectangle(
                panoramic_image, top_left, bottom_right, (255, 255, 255), cv2.FILLED
            )
            cv2.putText(
                panoramic_image,
                text,
                (start_col + 10, start_row + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # combine the query image with the panoramic view
        panoramic_image = np.hstack([query_image, panoramic_image])
        return panoramic_image

    def generate_ranking_explanation_azure_openai(
        self, caption: str, infos: dict
    ) -> tuple[list[int], str]:
        """
        uses an LLM to rank images and generate an explanation based on the combined panoramic view image, caption, and infos

        args:
            combined_image: the combined panoramic view image
            caption: description of the desired output images
            infos: related information about each image and the person

        returns:
            a tuple containing
                - a list of indices of the newly ranked images from most matched to least matched
                - an explanation from the LLM for the best item choice
        """
        base64_image = self.encode_image("combined_image.jpg")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        information = "You are responsible for ranking results for a Composed Image Retrieval. The user retrieves an image with an 'instruction' indicating their retrieval intent. For example, if the user queries a red car with the instruction 'change this car to blue,' a similar type of car in blue would be ranked higher in the results. Now you would receive instruction and query image with blue border.  Every item has its red index number in its top left. Do not misunderstand it. "
        information += f"User instruction: {caption} \n\n"
        # information += "The information in this picture is listed below, each with the corresponding index number in the image:\n\n"
        for i, info in enumerate(infos["product"]):
            information += f"{i}. {info}\n"

        information += "Provide a new ranked list of indices from most suitable to least suitable, followed by an explanation for the top 1 most suitable item only. The format of the response has to be 'Ranked list: []' with the indices in brackets as integers, followed by 'Reasons:' plus the explanation why this most fit user's query intent according to query image and query instruction."

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": information},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }
        response = self.client.chat.completions.create(
            model=self.azure_deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": information},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )
        response = json.loads(response.json())
        result = response["choices"][0]["message"]["content"]

        start_idx = result.find("[")
        end_idx = result.find("]")
        ranked_indices_str = result[start_idx + 1 : end_idx].split(",")
        ranked_indices = [int(index.strip()) for index in ranked_indices_str]

        # Extract explanation
        explanation = result[end_idx + 1 :].strip()

        return ranked_indices, explanation

    def generate_ranking_explanation(
        self, caption: str, infos: dict
    ) -> tuple[list[int], str]:
        """
        uses an LLM to rank images and generate an explanation based on the combined panoramic view image, caption, and infos

        args:
            caption: description of the desired output images
            infos: related information about each image and the person

        returns:
            a tuple containing
                - a list of indices of the newly ranked images from most matched to least matched
                - an explanation from the LLM for the best item choice
        """
        base64_image = self.encode_image("combined_image.jpg")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        information = (
            "You are responsible for ranking results for a Composed Image Retrieval. "
            "The user retrieves an image with an 'instruction' indicating their retrieval intent. "
            "For example, if the user queries a red car with the instruction 'change this car to blue,' a similar type of car in blue would be ranked higher in the results. "
            "Now you would receive instruction and query image with blue border. Every item has its red index number in its top left. Do not misunderstand it. "
            f"User instruction: {caption} \n\n"
        )

        # add additional information for each image
        for i, info in enumerate(infos["product"]):
            information += f"{i}. {info}\n"

        information += (
            "Provide a new ranked list of indices from most suitable to least suitable, followed by an explanation for the top 1 most suitable item only. "
            "The format of the response has to be 'Ranked list: []' with the indices in brackets as integers, followed by 'Reasons:' plus the explanation why this most fit user's query intent."
        )

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": information},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        result = response.json()["choices"][0]["message"]["content"]

        # parse the ranked indices from the response
        start_idx = result.find("[")
        end_idx = result.find("]")
        ranked_indices_str = result[start_idx + 1 : end_idx].split(",")
        ranked_indices = [int(index.strip()) for index in ranked_indices_str]

        # extract explanation
        explanation = result[end_idx + 1 :].strip()

        return ranked_indices, explanation

    def encode_image(self, image_path):
        """
        encodes the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def load_cache(self):
        """
        loads the cache from the file if it exists
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as file:
                return json.load(file)
        return {}

    def save_cache(self):
        """
        saves the current cache to the file cache.json
        """
        with open(self.cache_file, "w") as file:
            json.dump(self.cache, file)

    def generate_cache_key(self, caption, infos):
        """
        generates a unique cache key based on the caption and infos

        returns:
            a string that represents a unique cache key
        """
        return f"{caption}_{json.dumps(infos, sort_keys=True)}"
