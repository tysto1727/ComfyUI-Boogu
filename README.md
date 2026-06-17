# ComfyUI-Boogu

ComfyUI custom nodes for [Boogu-Image-0.1](https://github.com/boogu-project/Boogu-Image).

This repository only provides ComfyUI nodes and workflows. The inference package
and model weights come from the official Boogu project:

- Code: <https://github.com/boogu-project/Boogu-Image>
- Models: <https://huggingface.co/Boogu>
- License: Apache-2.0

## Supported Models

Download the official HuggingFace checkpoints with the same local directory names:

| Task | HuggingFace repo | ComfyUI loader |
| --- | --- | --- |
| Text-to-image | `Boogu/Boogu-Image-0.1-Base` | `BOOGU: Load Base Pipeline` |
| Image editing | `Boogu/Boogu-Image-0.1-Edit` | `BOOGU: Load Edit Pipeline` |
| Fast text-to-image | `Boogu/Boogu-Image-0.1-Turbo` | `BOOGU: Load Turbo Pipeline` |

Expected model layout:

```text
ComfyUI/models/boogu/
├── Boogu-Image-0.1-Base/
├── Boogu-Image-0.1-Edit/
└── Boogu-Image-0.1-Turbo/
```

## Installation

First install and verify the official Boogu-Image package following its README:

```bash
git clone https://github.com/boogu-project/Boogu-Image.git
cd Boogu-Image

conda create -y -n boogu python=3.10
conda activate boogu

pip install -r requirements/torch2.7-cu126.txt
pip install -e .
python utils/get_flash_attn.py
cd ..
```

Then install ComfyUI in the same environment:

```bash
git clone https://github.com/comfy-org/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

Install this custom node:

```bash
cd custom_nodes
git clone https://github.com/boogu-project/ComfyUI-Boogu.git
cd ..
```

Download the three official model checkpoints:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p models/boogu

hf download Boogu/Boogu-Image-0.1-Base \
  --local-dir models/boogu/Boogu-Image-0.1-Base

hf download Boogu/Boogu-Image-0.1-Edit \
  --local-dir models/boogu/Boogu-Image-0.1-Edit

hf download Boogu/Boogu-Image-0.1-Turbo \
  --local-dir models/boogu/Boogu-Image-0.1-Turbo
```

Start ComfyUI:

```bash
python main.py
```

## Workflows

The `workflows/` directory contains three minimal examples:

- `boogu_base_t2i.json`: `Load Base Pipeline -> Generate -> Preview`
- `boogu_edit_i2i.json`: `Load Edit Pipeline -> LoadImage -> Generate -> Preview`
- `boogu_turbo_t2i.json`: `Load Turbo Pipeline -> Generate -> Preview`

Prompt enhancement is intentionally not bundled in this repository. The Boogu
pipeline has an internal rewrite mechanism, but these ComfyUI nodes do not expose
or enable it. If you want prompt enhancement, connect an external ComfyUI Qwen or
LLM rewrite node to `BOOGU: Generate` via the optional `instruction_override`
input. When `instruction_override` is connected and non-empty, it replaces the
widget `instruction`.

## Node Notes

- `BOOGU: Load Base Pipeline` loads `Boogu-Image-0.1-Base`.
- `BOOGU: Load Edit Pipeline` loads `Boogu-Image-0.1-Edit`.
- `BOOGU: Load Turbo Pipeline` loads `Boogu-Image-0.1-Turbo`.
- `BOOGU: Generate` handles T2I, Edit, and Turbo generation.
- Turbo is pure T2I; do not connect `input_image` to a Turbo pipeline.
- For Turbo, use 3-4 steps, `text_guidance_scale=1.0`,
  `image_guidance_scale=1.0`, `empty_instruction_guidance_scale=0.0`,
  empty negative prompt, and `use_dmd_student_inference=True`.

## Troubleshooting

`ModuleNotFoundError: No module named 'boogu'`
: Install the official Boogu-Image package in the environment that launches
ComfyUI: `pip install -e /path/to/Boogu-Image`.

`ModuleNotFoundError: No module named 'sqlalchemy'`
: Install ComfyUI dependencies in the active environment:
`pip install -r requirements.txt`.

`cudart shared object not found`
: Check that your CUDA, PyTorch, and driver versions match the official
Boogu-Image environment. The tested setup is Python 3.10, CUDA 12.6, and
PyTorch 2.7.1.

`Resolved model path does not exist`
: Confirm the model directories are under `ComfyUI/models/boogu/` and keep the
official model directory names exactly:
`Boogu-Image-0.1-Base`, `Boogu-Image-0.1-Edit`, and
`Boogu-Image-0.1-Turbo`.
