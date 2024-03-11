from ultralytics import YOLO

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

    def predict(self, image: str | list) -> bool:  # func predict image
        try:
            # image predict
            self.model.predict(
                source=image,
                save=MODEL_PREDICT_SAVE_IMAGE,
                save_txt=MODEL_PREDICT_SAVE_TXT,
                imgsz=MODEL_IMAGE_SIZE,
                conf=MODEL_PREDICT_CONF,
            )
            return True

        except Exception as e:
            print(e)
            return False
