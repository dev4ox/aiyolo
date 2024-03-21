from typing import Annotated
import os

from fastapi import UploadFile, File, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from tools.api import APITools
from tools.os_custom import OSTools
from config import (
    API_VERSION,
    API_TITLE,
    API_DOCS_URL,
    API_OPEN_URL,
    DIR_IMAGES,
    DIR_RUNS,
    API_SERVER_HOST,
    API_SERVER_PORT,
)

_api_tools = APITools()

# get custom os tools
_os_tools = OSTools()

# init api app
_app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    openapi_url=API_OPEN_URL,
    docs_url=API_DOCS_URL,
)

# remove directories runs/ and images/
_os_tools.remove_file_or_directory(DIR_RUNS)
_os_tools.remove_file_or_directory(DIR_IMAGES)

# create directory images/
os.mkdir(DIR_IMAGES)

_app.mount("/static", StaticFiles(directory="images/"), name="static")


@_app.patch(
    "/api/patch_image",
    name="patch_image",
    description="На вход принимает изображение для обработки. "
                "Вернёт список распознаных классов в текстовом варианте.",
    response_class=JSONResponse,
)
async def get_image_txt(image: Annotated[UploadFile, File()]):
    data = await _api_tools.get_predict(image)

    return data


def start_local_server() -> None:  # func start local server to api
    # start uvicorn server
    uvicorn.run(
        app=_app,
        port=API_SERVER_PORT,
        host=API_SERVER_HOST
    )
