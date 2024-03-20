import os
import shutil


class OSTools:
    @staticmethod
    def remove_file_or_directory(path: str) -> None:  # func remove file or directory
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)

        elif os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
