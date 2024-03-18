from typing import Annotated
import os
import shutil
from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image

from config import (
    API_SERVER_PORT,
    API_TITLE,
    API_VERSION,
    API_DOCS_URL,
    API_OPEN_URL,
    DIR_IMAGES,
    FILE_TYPE_GET_IMAGE,
    DIR_RUNS,
    CLASSES,
    API_SERVER_HOST,
    MODEL_IMAGE_SIZE,
)
from model import Model


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    openapi_url=API_OPEN_URL,
    docs_url=API_DOCS_URL,
)


def _del_predict_path(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


_del_predict_path(DIR_RUNS)
_del_predict_path(DIR_IMAGES)
os.mkdir(DIR_IMAGES)
app.mount("/static", StaticFiles(directory="images/"), name="static")

model = Model()


def _image_file_resize(image_path: str) -> None:
    # get image in pillow object
    with Image.open(image_path) as image:
        image.load()

    new_size_image = (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    image_resize = image.resize(new_size_image)
    image_resize.save(image_path)
    image_resize.show()


def _create_image_file(img: Annotated[UploadFile, File()]) -> Path:
    # create dir to "get images"
    if not os.path.exists(DIR_IMAGES):
        os.mkdir(DIR_IMAGES)

    # path before "get image"
    image_path = DIR_IMAGES + str(img.file.name) + FILE_TYPE_GET_IMAGE
    with open(image_path, "wb") as image:
        image.write(img.file.read())

    # rename "get image"
    new_name_image = str(uuid.uuid4())
    new_path_image = DIR_IMAGES + new_name_image + FILE_TYPE_GET_IMAGE
    os.rename(image_path, new_path_image)

    # _image_file_resize(new_path_image)

    return Path(new_path_image)


async def _get_predict_image(image_path: Path) -> Path:
    # open predict image and get bytes of image
    new_path_output_image = DIR_IMAGES + image_path.name.split(image_path.suffix)[0] + "_output" + FILE_TYPE_GET_IMAGE
    shutil.copy(image_path, new_path_output_image)

    return Path(new_path_output_image)


async def _get_predict_text(image_path: Path) -> str:
    path_parent = str(image_path.parent)
    path_name = image_path.name
    path_suffix = image_path.suffix
    path_predict_txt = Path(path_parent + f"/labels/{path_name[:-len(path_suffix)]}.txt")
    predict_text = ""

    try:
        with open(path_predict_txt, "r") as predict_file_txt:
            data = predict_file_txt.readlines()

        for line in data:
            predict_text += CLASSES[int(line[:2])]

    except FileNotFoundError:
        pass

    return predict_text


async def _get_predict(image: Annotated[UploadFile, File()]) -> JSONResponse:
    # create image file from "get image"
    image_path = _create_image_file(image)

    # predict "get image"
    model.predict(str(image_path))
    image_path = Path(DIR_RUNS + f"detect/predict/" + Path(image_path).name)

    predict_text = await _get_predict_text(image_path)
    path_output_image = await _get_predict_image(image_path)

    # _del_predict_path(str(image_path.parent))

    json_response = {
        "predict text": predict_text,
        "image link": f"http://{API_SERVER_HOST}:{API_SERVER_PORT}/static/{path_output_image.name}"
    }

    # "http://127.0.0.1:1024/static/image.png"
    return JSONResponse(content=json_response)


@app.patch(
    "/api/patch_image",
    name="patch_image",
    description="На вход принимает изображение для обработки. "
                "Вернёт список распознаных классов в текстовом варианте.",
    response_class=JSONResponse,
)
async def get_image_txt(image: Annotated[UploadFile, File()]):
    data = await _get_predict(image)

    return data


def start_local_server() -> None:  # start local server to api
    uvicorn.run(app=app, port=API_SERVER_PORT, host=API_SERVER_HOST)
