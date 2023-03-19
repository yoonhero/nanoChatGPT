import torch

from model import GPTLanguageModel
from config import LARGE_GPT_CONFIG
from utils import load_model

model, _, _ = load_model("./tmp/checkpoints/epoch-80.tar", LARGE_GPT_CONFIG, best=False)
# model = GPTLanguageModel(LARGE_GPT_CONFIG)
X = torch.zeros((1, 128)).long()

input_names = ["Tokens"]
output_names = ["Next Token Prediction"]

torch.onnx.export(model, X, "./tmp/model.onnx", input_names=input_names, output_names=output_names)