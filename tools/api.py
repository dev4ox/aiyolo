import os
import shutil
from typing import Annotated
from pathlib import Path
import uuid

from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

from config import (
    DIR_IMAGES,
    DIR_RUNS,
    API_SERVER_HOST,
    API_SERVER_PORT,
    CLASSES,
    FILE_TYPE_GET_IMAGE,
    MODEL_IMAGE_SIZE,
)
from model import Model
from tools.pillow_custom import PillowTools
from singleton import Singleton


class APITools(metaclass=Singleton):
    def __init__(self):
        # init yolo model
        self._model = Model()

        # get custom pillow tools
        self._pillow_tools = PillowTools()

    def _input_image_file_crop(self, image_path: str, show_input_scaled_image: bool = False) -> None:
        # get object pillow image
        image = self._pillow_tools.open(image_path)
        new_image_size = (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)

        # get size image
        width, height = image.size

        # Determining the dimensions for cropping and scaling
        new_width = min(width, new_image_size[0])
        new_height = min(height, new_image_size[1])

        # Crop the image
        left = max(0, (width - new_width) // 2)
        top = max(0, (height - new_height) // 2)
        right = min(width, left + new_width)
        bottom = min(height, top + new_height)
        cropped_image = image.crop((left, top, right, bottom))

        # save input scaled image
        cropped_image.save(image_path)

        # show input scaled image
        if show_input_scaled_image:
            cropped_image.show()

    def _create_image_file(self, input_image: Annotated[UploadFile, File()]) -> Path:  # func create input image
        # create directory to input image
        if not os.path.exists(DIR_IMAGES):
            os.mkdir(DIR_IMAGES)

        # create path to input image
        path_input_image = DIR_IMAGES + str(input_image.file.name) + FILE_TYPE_GET_IMAGE

        # create input image file and write input image bytes
        with open(path_input_image, "wb") as input_image_file:
            input_image_file.write(input_image.file.read())

        # create new name to input image file and create path with new name input image
        new_name_image = str(uuid.uuid4())
        new_path_image = DIR_IMAGES + new_name_image + FILE_TYPE_GET_IMAGE

        # rename input image file
        os.rename(path_input_image, new_path_image)

        # crop input image
        self._input_image_file_crop(new_path_image)

        return Path(new_path_image)

    @staticmethod
    async def _get_predict_image(image_path: Path) -> Path:  # func get predict image
        # open predict image and get bytes of image
        new_path_output_image = (DIR_IMAGES + image_path.name.split(image_path.suffix)[0] +
                                 "_output" + FILE_TYPE_GET_IMAGE)

        # copy predict image
        shutil.copy(image_path, new_path_output_image)

        return Path(new_path_output_image)

    @staticmethod
    def _sort_predict_text(predict_file_data: list) -> list:  # func sort predict text
        # Create a list to store data
        detections = []

        # We split the lines and save the data into a list
        for line in predict_file_data:
            parts = line.strip().split()
            label = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            detections.append((label, x_center, y_center, width, height))

        # Sort the list by object center coordinates
        detections.sort(key=lambda x: (x[2], x[1]))  # Sort by y_center and then by x_center

        # Output sorted data
        for detection in detections:
            print(detection)

        return detections

    async def _get_predict_text(self, image_path: Path) -> str:  # func get predict text
        # get paths from "image_path"
        path_parent = str(image_path.parent)
        path_name = image_path.name
        path_suffix = image_path.suffix

        # create path before predict text file
        path_predict_txt = Path(path_parent + f"/labels/{path_name[:-len(path_suffix)]}.txt")
        predict_text = ""

        try:
            # open predict text file and get data from predict text file
            with open(path_predict_txt, "r") as predict_file_txt:
                data = predict_file_txt.readlines()

            # sort data from predict text file
            data = self._sort_predict_text(data)

            # get classes from predict text data
            for line in data:
                predict_text += CLASSES[line[0]]

        except FileNotFoundError:
            pass

        return predict_text

    # async func get predict image and predict text
    async def get_predict(self, image: Annotated[UploadFile, File()]) -> JSONResponse:
        # create image file from "get image"
        image_path = self._create_image_file(image)

        # predict "get image"
        self._model.predict(str(image_path))

        # create path image
        image_path = Path(DIR_RUNS + f"detect/predict/" + Path(image_path).name)

        # get predict image and predict text
        predict_text = await self._get_predict_text(image_path)
        path_output_image = await self._get_predict_image(image_path)

        # create json response
        json_response = {
            "predict text": predict_text,
            "image link": f"http://{API_SERVER_HOST}:{API_SERVER_PORT}/static/{path_output_image.name}"
        }

        return JSONResponse(content=json_response)
