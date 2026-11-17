"""Image preparation and reusable model inference."""

import gc
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageOps


def read_image(path: Path) -> Image.Image:
    """Apply camera orientation and detach pixels from the source file."""
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        # Composite transparent inputs onto white rather than exposing hidden RGB.
        if oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return oriented.convert("RGB")


def prepare_image(image: Image.Image, max_edge: int) -> Image.Image:
    result = image.copy()
    if max_edge:
        result.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return result


class RestorationEngine:
    """Keep the current restoration model loaded between runs."""

    def __init__(self):
        self.model = None
        self.key = None
        self.auto_devices = {}

    def load(self, checkpoint, device, report):
        import torch

        stat = checkpoint.stat()
        key = (str(checkpoint.resolve()), stat.st_mtime_ns, stat.st_size)
        if self.key != key:
            self.model = None
            self.key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report(f"Loading model on {device.upper()}…")
            self.model = self.load_model(checkpoint, device)
            self.key = key
        else:
            self.model.to(device)

    def load_model(self, checkpoint, device):
        from jpeg_model import load_jpeg_model
        return load_jpeg_model(str(checkpoint), device=device)

    def predict(self, pixels, steps):
        from jpeg_model import restore
        return restore(self.model, pixels, sample_steps=steps)

    def input_tensor(self, image, device):
        import torch
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(image.height, image.width, 3)
        return pixels.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255

    def output_image(self, output):
        import torch
        output = output[0].clamp(0, 1).mul(255).round().to(torch.uint8)
        output = output.permute(1, 2, 0).contiguous().cpu()
        return Image.frombytes("RGB", (output.shape[1], output.shape[0]), output.numpy().tobytes())

    def select_auto_device(self, checkpoint, image, steps, report):
        """Compare actual restoration on a small preview; cache per workload/checkpoint."""
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        candidates = [f"cuda:{index}" for index in range(torch.cuda.device_count())
                      if torch.cuda.get_device_properties(index).total_memory >= 2 * 1024**3]
        if not candidates:
            report("Auto: CPU selected; no CUDA GPU has at least 2 GB VRAM.")
            return "cpu"
        preview = prepare_image(image, 128)
        stat = checkpoint.stat()
        key = (str(checkpoint.resolve()), stat.st_mtime_ns, stat.st_size, preview.size, steps)
        if key in self.auto_devices:
            return self.auto_devices[key]

        self.load(checkpoint, "cpu", report)
        probe = torch.full((1, 3, preview.height, preview.width), 0.5)

        report("Auto: comparing CPU and GPU restoration speed (first use)…")
        best_device, best_time = "cpu", self.benchmark_device("cpu", probe, steps)
        for device in candidates:
            try:
                elapsed = self.benchmark_device(device, probe, steps)
            except RuntimeError:
                # An incompatible or memory-constrained GPU must not prevent CPU use.
                self.model.to("cpu")
                torch.cuda.empty_cache()
                continue
            if elapsed <= best_time:
                best_device, best_time = device, elapsed
        self.auto_devices[key] = best_device
        report(f"Auto selected {best_device.upper()} using a {preview.width} × {preview.height} restoration benchmark.")
        return best_device

    def benchmark_device(self, device, probe, steps):
        import torch

        self.model.to(device)
        pixels = probe.to(device)
        gpu = device.startswith("cuda")
        devices = [torch.device(device).index] if gpu else []
        timings = []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            # Warm up kernels, then include input/output transfers in each timed run.
            self.predict(pixels, steps).cpu()
            for _ in range(2):
                if gpu:
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                self.predict(probe.to(device), steps).cpu()
                if gpu:
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - start)
        return statistics.median(timings)

    def run(self, image, checkpoint, device, steps, seed, report):
        import torch

        if device == "Auto":
            device = self.select_auto_device(checkpoint, image, steps, report)
        else:
            device = device.lower()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(
                "This Python environment cannot use CUDA. Install a CUDA-enabled "
                "PyTorch build and confirm that a compatible GPU is available.\n\n"
                f"Python: {sys.executable}\nPyTorch: {torch.__version__}\n"
                f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}\n\n"
                "Auto or CPU can also run without CUDA."
            )

        self.load(checkpoint, device, report)

        report(f"Restoring {image.width} × {image.height} on {device.upper()}…")
        # Avoid importing torchvision for a simple RGB byte-to-tensor conversion.
        pixels = self.input_tensor(image, device)
        # Isolate the seed so repeat runs do not modify the caller's RNG state.
        gpu = device.startswith("cuda")
        gpu_index = (torch.device(device).index if ":" in device else torch.cuda.current_device()) if gpu else None
        devices = [gpu_index] if gpu else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if gpu:
                with torch.cuda.device(gpu_index):
                    torch.cuda.manual_seed(seed)
            output = self.predict(pixels, steps)
        if not torch.isfinite(output).all().item():
            raise RuntimeError("The model produced non-finite pixels. Try CPU or another checkpoint.")
        return self.output_image(output), device
