"""Download the exact FBCNN checkpoints used by KenjieDec's Hugging Face Space."""

import argparse
import hashlib
from pathlib import Path

import requests


REVISION = "5eee4c8c4f9c1606ac1463b65b4d9da178131cf9"
CHECKPOINTS = {
    "fbcnn_color.pth": "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
    "fbcnn_gray.pth": "7beec25b883a6c3bd7d103429b78a2371f7e85c1f5261a6e2e5d0702d9e9f977",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(name):
    destination = OUTPUT_DIR / name
    expected = CHECKPOINTS[name]
    if destination.is_file() and file_hash(destination) == expected:
        print(f"Verified {name}", flush=True)
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    url = f"https://huggingface.co/spaces/KenjieDec/FBCNN/resolve/{REVISION}/{name}"
    print(f"Downloading {name} (about 288 MB)…", flush=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    if file_hash(temporary) != expected:
        raise RuntimeError(f"SHA-256 mismatch for {name}; download was not installed.")
    temporary.replace(destination)
    print(f"Verified and installed {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-only", action="store_true")
    args = parser.parse_args()
    for name in CHECKPOINTS:
        if not args.color_only or name == "fbcnn_color.pth":
            download(name)


if __name__ == "__main__":
    main()
