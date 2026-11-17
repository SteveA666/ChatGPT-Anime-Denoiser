"""Local FBCNN inference matching the pinned KenjieDec/FBCNN Space."""

import argparse
from pathlib import Path

from restoration_engine import RestorationEngine, read_image


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class FBCNNEngine(RestorationEngine):
    def __init__(self, grayscale=False, intensity=40):
        super().__init__()
        self.grayscale = grayscale
        self.intensity = intensity

    def load_model(self, checkpoint, device):
        import torch
        from third_party.fbcnn.network_fbcnn import FBCNN

        channels = 1 if self.grayscale else 3
        model = FBCNN(in_nc=channels, out_nc=channels, nc=[64, 128, 256, 512], nb=4, act_mode="R")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        return model.eval().requires_grad_(False).to(device)

    def predict(self, pixels, steps):
        import torch

        # Its two quality inversions simplify to qf_input = slider_value / 100.
        if not 1 <= self.intensity <= 100:
            raise ValueError("FBCNN intensity must be between 1 and 100.")
        quality = torch.tensor([[self.intensity / 100]], dtype=pixels.dtype, device=pixels.device)
        restored, _ = self.model(pixels, quality)
        return restored

    def input_tensor(self, image, device):
        import numpy as np
        import torch

        # Match the Space's CPU normalization, tensor layout, and BGR conversion.
        source = np.array(image.convert("L"))[:, :, None] if self.grayscale else np.array(image)[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(source)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    def output_image(self, output):
        import numpy as np
        from PIL import Image

        data = output[0].permute(1, 2, 0).float().cpu().numpy()
        data = np.uint8(np.round(np.clip(data, 0, 1) * 255))
        return (Image.fromarray(data[:, :, 0]).convert("RGB") if self.grayscale
                else Image.fromarray(np.ascontiguousarray(data[:, :, ::-1])))

    def benchmark_device(self, device, probe, steps):
        return super().benchmark_device(device, probe[:, :1] if self.grayscale else probe, steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--intensity", type=int, default=40,
                        help="Same numeric value as the Space's Intensity slider (default 40)")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--device", choices=("Auto", "CUDA", "CPU"), default="Auto")
    args = parser.parse_args()
    engine = FBCNNEngine(args.grayscale, args.intensity)
    checkpoint = CHECKPOINT_DIR / ("fbcnn_gray.pth" if args.grayscale else "fbcnn_color.pth")
    if not checkpoint.is_file():
        parser.error("Checkpoint missing. Run python setup_fbcnn.py first.")
    output, _ = engine.run(read_image(args.image), checkpoint, args.device, 1, 0, print)
    destination = args.output or args.image.with_name(f"{args.image.stem}_fbcnn.png")
    output.save(destination)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
