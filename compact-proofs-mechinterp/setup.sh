#!/bin/sh

uv init -p 3.12

echo '
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true' >>pyproject.toml

uv add marimo watchdog basedpyright ty
uv add einops jaxtyping matplotlib torch tqdm
