import torch
from safetensors import safe_open
from pathlib import Path

def convert_safetensor_to_pt(input_path):
    input_path = Path(input_path)
    if input_path.suffix != ".safetensors":
        raise ValueError("Input file must have a .safetensors extension")

    tensors = {}
    with safe_open(input_path.as_posix(), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    output_path = input_path.with_suffix(".pt")

    torch.save(tensors, output_path)
    print(f"Converted and saved: {output_path}")

input_file = "./ckpts/models--SWivid--F5-TTS/snapshots/4dcc16f297f2ff98a17b3726b16f5de5a5e45672/F5TTS_Base/model_1200000.safetensors"
convert_safetensor_to_pt(input_file)
