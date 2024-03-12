import os

# dirs
CDIR = os.getcwd() + "/"
DIR_IMAGES = CDIR + "images/"
DIR_RUNS = CDIR + "runs/"

# model
MODEL_IMAGE_SIZE = 480
MODEL_TRAIN_EPOCHS = 10
MODEL_IMAGE_BATCH = 8
MODEL_NAME = 'ds1'
MODEL_PREDICT_SAVE_IMAGE = True
MODEL_PREDICT_SAVE_TXT = True
MODEL_PREDICT_CONF = 0.5

# api
API_SERVER_PORT = 1024
API_TITLE = "AiYOLO-API"
API_VERSION = "0.0.1"
API_OPEN_URL = "/aiYOLO-api.json"
API_DOCS_URL = "/"

# file names
FILE_NAME_CONFIG_TRAIN_MODEL = "data"

# file types
FILE_TYPE_CONFIG_TRAIN_MODEL = ".yaml"
FILE_TYPE_GET_IMAGE = ".png"

# classes
CLASSES = {  # classes num 21
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'E',
    14: 'H',
    15: 'K',
    16: 'M',
    17: 'O',
    18: 'P',
    19: 'T',
    20: 'X',
    21: 'Y',
}
