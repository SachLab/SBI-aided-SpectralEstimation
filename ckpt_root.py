import os

_HERE = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.environ.get("CKPT_DIR", "artifacts_total")
ARTIFACTS = (CKPT_DIR if os.path.isabs(CKPT_DIR)
             else os.path.join(_HERE, CKPT_DIR))

# ARTIFACTS_LEGACY = os.path.join(_HERE, "artifacts")


def ckpt_path(*parts):
    return os.path.join(ARTIFACTS, *parts)

TB_ROOT = os.environ.get("TB_DIR", os.path.join(_HERE, "tf-logs"))


def no_clobber(path):
    path = os.path.normpath(path)
    taken = (os.path.isfile(path)
             or (os.path.isdir(path) and os.listdir(path)))
    if not taken:
        return path
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    root, ext = (os.path.splitext(path) if os.path.isfile(path) else (path, ""))
    new = f"{root}_{ts}{ext}"
    return new
