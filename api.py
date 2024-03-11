from typing import Annotated
import os
import glob
from pathlib import Path

from fastapi import FastAPI, UploadFile, Response, File
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
)
from model import Model


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    openapi_url=API_OPEN_URL,
    docs_url=API_DOCS_URL,
)

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


@app.patch("/api/patch_image",
           name="patch_image",
           description="drg",
           responses={
               200: {
                   "content": {"image/png": {}}
               }
           },
           # response_class=Response
           )
async def get_image(img: Annotated[UploadFile, File()]):  # get image to predict
    # create image file from "get image"
    image_path = _create_image_file(img)

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
        with open(DIR_RUNS + f"detect/predict{now_predict_num}/" + Path(image_path).name, "rb") as image:
            image_bytes_return = image.read()

        return Response(content=image_bytes_return, media_type="image/png")

    else:
        raise


def start_local_server(server_port: int | None = None) -> None:  # start local server to api
    if server_port is None:
        uvicorn.run(app=app, port=API_SERVER_PORT)

    elif server_port is not None:
        uvicorn.run(app=app, port=server_port)
