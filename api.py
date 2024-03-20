from typing import Annotated

from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

from tools.api import APITools

api_tools = APITools()


@api_tools.app.patch(
    "/api/patch_image",
    name="patch_image",
    description="На вход принимает изображение для обработки. "
                "Вернёт список распознаных классов в текстовом варианте.",
    response_class=JSONResponse,
)
async def get_image_txt(image: Annotated[UploadFile, File()]):
    data = await api_tools.get_predict(image)

    return data
