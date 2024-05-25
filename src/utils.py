import joblib
import yaml


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("value and filename must not be None".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)
    else:
        raise ValueError("filename must not be None".capitalize())


def config():
    with open("./config.yml", "r") as f:
        return yaml.safe_load(f)
