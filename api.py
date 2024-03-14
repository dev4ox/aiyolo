from typing import Annotated
import os
import glob
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

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
)
from model import Model


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    openapi_url=API_OPEN_URL,
    docs_url=API_DOCS_URL,
)
app.mount("/static", StaticFiles(directory="images/"), name="static")

model = Model()


def _create_image_file(img: Annotated[UploadFile, File()]) -> Path:
    # create dir to "get images"
    if not os.path.exists(DIR_IMAGES):
        os.mkdir(DIR_IMAGES)

    # path before "get image"
    image_path = DIR_IMAGES + str(img.file.name) + FILE_TYPE_GET_IMAGE
    with open(image_path, "wb") as image:
        image.write(img.file.read())

    # rename "get image"
    os.rename(image_path, DIR_IMAGES + "image" + FILE_TYPE_GET_IMAGE)
    image_path = DIR_IMAGES + "image" + FILE_TYPE_GET_IMAGE

    return Path(image_path)


async def _get_predict_image(image_path: Path) -> bytes:
    # open predict image and get bytes of image
    with open(image_path, "rb") as image_file:
        image_bytes_return = image_file.read()

    return image_bytes_return


async def _get_predict_text(image_path: Path) -> str:
    path_parent = str(image_path.parent)
    path_name = image_path.name
    path_suffix = image_path.suffix
    path_predict_txt = Path(path_parent + f"/labels/{path_name[:-len(path_suffix)]}.txt")
    predict_text = ""

    with open(path_predict_txt, "r") as predict_file_txt:
        data = predict_file_txt.readlines()

    for line in data:
        predict_text += CLASSES[int(line[:2])]

    return predict_text


async def _get_predict(image: Annotated[UploadFile, File()]) -> HTMLResponse:
    # create image file from "get image"
    image_path = _create_image_file(image)

    # predict "get image"
    model.predict(str(image_path))
    image_path = Path(DIR_RUNS + f"detect/predict/" + Path(image_path).name)

    predict_text = await _get_predict_text(image_path)
    image_bytes = await _get_predict_image(image_path)

    _del_predict_path(str(image_path.parent))

    html_response = """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Look ma! HTML!</h1>
        </body>
    </html>
    """

    # "http://127.0.0.1:1024/static/image.png"
    return HTMLResponse(content=html_response)


def _del_predict_path(path_now_predict: str) -> None:
    for path in glob.glob(path_now_predict + "*"):
        if os.path.isfile(path):
            os.remove(path)

        elif os.path.isdir(path):
            try:
                os.removedirs(path)

            except OSError:
                _del_predict_path(path + "/")
                os.removedirs(path)


@app.patch(
    "/api/patch_image",
    name="patch_image",
    description="На вход принимает изображение для обработки. "
                "Вернёт список распознаных классов в текстовом варианте.",
    response_class=HTMLResponse,
)
async def get_image_txt(image: Annotated[UploadFile, File()]) -> HTMLResponse:
    data = await _get_predict(image)

    return data


def start_local_server(server_port: int | None = None) -> None:  # start local server to api
    if server_port is None:
        uvicorn.run(app=app, port=API_SERVER_PORT)

    elif server_port is not None:
        uvicorn.run(app=app, port=server_port)
