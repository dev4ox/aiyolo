from model import Model
from api import *


class Main:
    def __init__(self):
        # https://habr.com/ru/articles/714232/

        # yolo model init in main class
        self.model = Model()
        print("model init done")


if __name__ == '__main__':
    main = Main()
    start_local_server()
    # main.model.train()

