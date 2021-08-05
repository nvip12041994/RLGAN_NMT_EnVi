import importlib
import os


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        model_name = file[: file.find(".py")]
        importlib.import_module("RLGAN_NMT_EnVi.models." + model_name)