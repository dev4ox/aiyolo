from PIL import Image


class PillowTools:
    @staticmethod
    def open(path: str):  # func get object pillow image
        with Image.open(path) as image:
            image.load()

        return image
