from ultralytics import YOLO
from PIL import Image

from config import (
    MODEL_TRAIN_EPOCHS,
    MODEL_NAME,
    MODEL_IMAGE_SIZE,
    MODEL_IMAGE_BATCH,
    FILE_NAME_CONFIG_TRAIN_MODEL,
    FILE_TYPE_CONFIG_TRAIN_MODEL,
    MODEL_PREDICT_SAVE_IMAGE,
    MODEL_PREDICT_CONF,
    MODEL_PREDICT_SAVE_TXT,
)


class Model:
    def __init__(self):
        # init model
        self.model = YOLO('best_model_new_ds.pt')

    def train(self) -> dict | None:  # func train model
        # train model start
        results = self.model.train(
            data=FILE_NAME_CONFIG_TRAIN_MODEL + FILE_TYPE_CONFIG_TRAIN_MODEL,
            imgsz=MODEL_IMAGE_SIZE,
            epochs=MODEL_TRAIN_EPOCHS,
            batch=MODEL_IMAGE_BATCH,
            name=MODEL_NAME
        )

        return results

    # @staticmethod
    # def _normalization_size_image(size: tuple[int, int]) -> tuple[int, int]:
    #     # get nums multiple 32
    #     list_nums_multiple_32 = []
    #     num = 32
    #
    #     while True:
    #         if num >= 1_000_000:
    #             break
    #
    #         list_nums_multiple_32.append(num)
    #         num *= 2
    #
    #     print(list_nums_multiple_32)
    #
    #     # check multiple 32 on image size
    #     size = list(size)
    #     print(size)
    #     print(size[1] % 32)
    #     if size[0] % 32 != 0:
    #         for num in list_nums_multiple_32:
    #             for num1 in list_nums_multiple_32:
    #                 if num < size[0] < num1:
    #                     size[0] = num1
    #                     break
    #
    #     elif size[1] % 32 != 0:
    #         for num in list_nums_multiple_32:
    #             for num1 in list_nums_multiple_32:
    #                 if num < size[1] < num1:
    #                     size[1] = num1
    #                     break
    #
    #     elif size[0] % 32 == 0 and size[1] % 32 == 0:
    #         size = tuple(size)
    #
    #     print(size)
    #     return size
    def predict(self, image: str | list) -> None:  # func predict image
        # get image size
        with Image.open(image) as image_file:
            image_file.load()

        # image_size_normalization = self._normalization_size_image(image_file.size)
        image_file.show()

        # image predict
        self.model.predict(
            source=image,
            save=MODEL_PREDICT_SAVE_IMAGE,
            save_txt=MODEL_PREDICT_SAVE_TXT,
            imgsz=MODEL_IMAGE_SIZE,  # MODEL_IMAGE_SIZE
            conf=MODEL_PREDICT_CONF,
        )
