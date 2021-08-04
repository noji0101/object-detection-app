#@title デフォルトのタイトル テキスト
from pathlib import Path


class Errors:
    LINE = "=" * 25
    BASE_MSG = "\n{line}\n".format(line=LINE)

    def __call__(self, msg, exception):
        return exception(msg)

    @classmethod
    def FileNotFound(cls, path: Path):
        path = Path(path)
        msg = cls.BASE_MSG
        msg += "NOT Exists Path: {}\n".format(path)

        path_gradually = Path(path.parts[0]).absolute()
        for path_part in path.parts[1:]:
            path_gradually /= path_part
            msg += "\tExists: {}, {}\n".format(path_gradually.exists(), path_gradually)

        return FileNotFoundError(msg)
