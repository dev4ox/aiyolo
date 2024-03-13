from typing import Annotated
import os
import glob
from pathlib import Path
import random
import gc

from fastapi import FastAPI, UploadFile, Response, File
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
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


class CustomHTTPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Перед вызовом следующего обработчика, устанавливаем параметры соединения
        request.scope["http_connection"] = "close"  # Отключаем Keep-Alive
        response = await call_next(request)
        return response


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    openapi_url=API_OPEN_URL,
    docs_url=API_DOCS_URL,
)


@app.middleware("http")
async def secure_middleware(request, call_next):
    response = await call_next(request)
    del response.headers["Server"]
    del response.headers["server"]
    return response

model = Model()


def _create_image_file(img: Annotated[UploadFile, File()]) -> str:
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

    return image_path


def _get_predict_path(image: Annotated[UploadFile, File()]) -> str:
    # create image file from "get image"
    image_path = _create_image_file(image)

    # predict "get image"
    predict_done = model.predict(image_path)

    # success check predict
    if predict_done:
        predict_paths = []
        dir_nums = []
        runs_paths = glob.glob(DIR_RUNS + "detect/*")

        for directory in runs_paths:
            for i in range(len(runs_paths)):
                if f"predict{i if i != 0 else ''}" in directory:
                    dir_nums.append(i)
                    predict_paths.append(directory)

        now_predict_num = max(dir_nums)
        return DIR_RUNS + f"detect/predict{now_predict_num}/" + Path(image_path).name

    else:
        raise


@app.patch(
    "/api/patch_image",
    name="patch_image",
    description="На вход принимает изображение для обработки. "
                "Вернёт обработано изображение с выделными классами, которые были распознаны.",
    responses={
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=Response
)
def get_image(image: Annotated[UploadFile, File()]) -> Response:  # get image to predict
    # get now predict path
    now_predict_path = _get_predict_path(image)

    # open predict image and get bytes of image
    with open(now_predict_path, "rb") as image_file:
        image_bytes_return = image_file.read()

    return Response(content=image_bytes_return, media_type="image/png")

@app.patch(
    "/api/patch_image_txt",
    name="patch_image_txt",
    description="На вход принимает изображение для обработки. "
                "Вернёт список распознаных классов в текстовом варианте.",
)
async def get_image_txt(image: Annotated[UploadFile, File()]) -> Response:
    now_predict_path = Path(_get_predict_path(image))
    path_parent = str(now_predict_path.parent)
    path_name = now_predict_path.name
    path_suffix = now_predict_path.suffix
    text_temp = []

    with open(path_parent + f"/labels/{path_name[:-len(path_suffix)]}.txt", "r") as predict_file_txt:
        text_temp.clear()
        data = predict_file_txt.readlines()
    for line in data:
        text_temp.append(CLASSES[int(line[:2])])
    headers = {
        'Cache-Control': f'no-cache, no-store, must-revalidate, {random.random()}',
        'Pragma': 'no-cache'
    }
    response = Response(content="".join(text_temp))
    response.headers["Connection"] = "close"
    return response
# async def get_image_txt(image: Annotated[UploadFile, File()]) -> str:
#     now_predict_path = Path(await _get_predict_path(image))
#     path_parent = str(now_predict_path.parent)
#     path_name = now_predict_path.name
#     path_suffix = now_predict_path.suffix
#
#     text_temp = ""
#     with open(path_parent + f"/labels/{path_name[:-len(path_suffix)]}.txt", "r") as predict_file_txt:
#         data = list(predict_file_txt.read())
#         count_string = 0
#         strings_dict = []
#
#         for value in data:
#             text_temp += value
#
#             if value == "\n":
#                 strings_dict.append(text_temp)
#                 text_temp = ""
#                 count_string += 1
#
#         text_temp = ""
#         for string in strings_dict:
#             text_temp += CLASSES[int(list(string)[0] + list(string)[1])]
#
#         print(text_temp)
#
#     return text_temp


def start_local_server(server_port: int | None = None) -> None:  # start local server to api
    if server_port is None:
        uvicorn.run(app=app, port=API_SERVER_PORT)

    elif server_port is not None:
        uvicorn.run(app=app, port=server_port)
