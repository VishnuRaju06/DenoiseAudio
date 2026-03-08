import torch

try:
    model = torch.load('best_model.pth', map_location='cpu')
    print("Type of loaded object:", type(model))
    if isinstance(model, dict):
        print("Keys in dict:", model.keys())
except Exception as e:
    print("Error loading model:", e)
