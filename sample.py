import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from PIL import Image

from jpeg_model import load_jpeg_model, restore

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_jpeg_model("meanflow_jpeg_qf5_20.pth", device)

to_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore a JPEG-degraded image with MeanFlow")
    parser.add_argument("input", type=Path, help="path to the degraded JPEG image")
    parser.add_argument("--output", type=Path, default=None, help="output path (default: <input>_restored.png)")
    parser.add_argument("--steps", type=int, default=2, choices=(1, 2), help="MeanFlow sampling steps")
    args = parser.parse_args()

    input_img = Image.open(args.input).convert("RGB")
    degraded = to_tensor(input_img).unsqueeze(0).to(device)

    restored = restore(model, degraded, sample_steps=args.steps)

    output = args.output or args.input.with_name(f"{args.input.stem}_restored.png")
    v2.functional.to_pil_image(restored[0].cpu()).save(output)
    print(f"saved {output}")
