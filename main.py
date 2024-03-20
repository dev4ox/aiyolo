from model import Model
from tools.api import APITools


class Main:
    def __init__(self):
        # https://habr.com/ru/articles/714232/

        # yolo model init in main class
        self.model = Model()
        print("model init done")

        # api tools init in main class
        self.api_tools = APITools()
        print("api tools init done")


if __name__ == '__main__':
    main = Main()
    main.api_tools.start_local_server()
    # main.model.train()
