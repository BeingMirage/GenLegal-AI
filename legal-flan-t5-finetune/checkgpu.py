import torch

import torch

print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

