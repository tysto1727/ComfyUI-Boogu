"""ComfyUI custom nodes for the public Boogu-Image pipelines."""

from __future__ import annotations

import gc
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths  # provided by ComfyUI at runtime

    MODELS_ROOT = os.path.join(folder_paths.models_dir, "boogu")
except Exception:
    folder_paths = None
    MODELS_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "boogu",
    )

BASE_MODEL_NAME = "Boogu-Image-0.1-Base"
EDIT_MODEL_NAME = "Boogu-Image-0.1-Edit"
TURBO_MODEL_NAME = "Boogu-Image-0.1-Turbo"

DEFAULT_NEGATIVE_INSTRUCTION = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
    "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
    "(poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
)

DEVICE_CHOICES = ["cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7", "cpu"]
DTYPE_CHOICES = ["bf16", "fp16", "fp32"]
PIPELINE_CACHE: dict[tuple, object] = {}


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _import_boogu_pipelines():
    try:
        from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
        from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import the `boogu` package. Install Boogu-Image first, "
            "for example: `pip install -e /path/to/Boogu-Image`."
        ) from exc
    return BooguImagePipeline, BooguImageTurboPipeline


def _canonical_device(device: str) -> str:
    return "cuda:0" if device == "cuda" else str(device)


def _resolve_model_path(model_name: str) -> str:
    path = os.path.join(MODELS_ROOT, model_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Resolved model path does not exist: {path}\n"
            f"Download Boogu/{model_name} to ComfyUI/models/boogu/{model_name}."
        )
    return path


def _cache_key(kind: str, model_path: str, dtype: str, device: str, offload: bool) -> tuple:
    return (kind, os.path.realpath(model_path), dtype, _canonical_device(device), bool(offload))


def _configure_pipeline_device(pipeline, device: str, enable_model_cpu_offload: bool):
    pipeline._boogu_device = _canonical_device(device)
    pipeline.enable_model_cpu_offload_flag = bool(enable_model_cpu_offload)
    pipeline.enable_sequential_cpu_offload_flag = False
    pipeline.enable_group_offload_flag = False
    os.environ["device"] = pipeline._boogu_device
    if enable_model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    elif pipeline._boogu_device is not None:
        pipeline = pipeline.to(pipeline._boogu_device)
    return pipeline


def _load_pipeline(kind: str, model_name: str, dtype: str, device: str, enable_model_cpu_offload: bool, force_reload: bool):
    BooguImagePipeline, BooguImageTurboPipeline = _import_boogu_pipelines()
    model_path = _resolve_model_path(model_name)
    key = _cache_key(kind, model_path, dtype, device, enable_model_cpu_offload)

    if force_reload and key in PIPELINE_CACHE:
        print("[BOOGU] force_reload=True, dropping cached pipeline")
        del PIPELINE_CACHE[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if key not in PIPELINE_CACHE:
        PipelineClass = BooguImageTurboPipeline if kind == "turbo" else BooguImagePipeline
        print(f"[BOOGU] Loading {kind}: {model_path} dtype={dtype} device={device} offload={enable_model_cpu_offload}")
        pipe = PipelineClass.from_pretrained(model_path, torch_dtype=_resolve_dtype(dtype), trust_remote_code=True)
        pipe._boogu_kind = kind
        pipe._boogu_model_name = model_name
        pipe._boogu_model_path = model_path
        PIPELINE_CACHE[key] = _configure_pipeline_device(pipe, device, enable_model_cpu_offload)
        print("[BOOGU] Pipeline ready.")
    else:
        print("[BOOGU] Reusing cached pipeline.")
    return (PIPELINE_CACHE[key],)


def _pil_from_image_tensor(image: torch.Tensor) -> list[Image.Image]:
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")
    arr = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return [Image.fromarray(arr[i]).convert("RGB") for i in range(arr.shape[0])]


def _image_tensor_from_pil(pil_images: list[Image.Image]) -> torch.Tensor:
    arrs = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrs, axis=0))


class _BOOGULoader:
    MODEL_NAME = ""
    KIND = ""
    DISPLAY_KIND = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (DTYPE_CHOICES, {"default": "bf16"}),
                "device": (DEVICE_CHOICES, {"default": "cuda"}),
                "enable_model_cpu_offload": ("BOOLEAN", {"default": False}),
            },
            "optional": {"force_reload": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("BOOGU_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "BOOGU/loaders"

    def load(self, dtype, device, enable_model_cpu_offload, force_reload=False):
        print(f"[BOOGU] Using fixed {self.DISPLAY_KIND} model: {self.MODEL_NAME}")
        return _load_pipeline(self.KIND, self.MODEL_NAME, dtype, device, enable_model_cpu_offload, bool(force_reload))


class BOOGULoadBasePipeline(_BOOGULoader):
    MODEL_NAME = BASE_MODEL_NAME
    KIND = "base"
    DISPLAY_KIND = "Base T2I"


class BOOGULoadEditPipeline(_BOOGULoader):
    MODEL_NAME = EDIT_MODEL_NAME
    KIND = "edit"
    DISPLAY_KIND = "Edit I2I"


class BOOGULoadTurboPipeline(_BOOGULoader):
    MODEL_NAME = TURBO_MODEL_NAME
    KIND = "turbo"
    DISPLAY_KIND = "Turbo T2I"


class BOOGUGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("BOOGU_PIPELINE",),
                "instruction": ("STRING", {"default": "画一头在森林里抓拍到的棕熊", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "text_guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "image_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "empty_instruction_guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "num_images_per_instruction": ("INT", {"default": 1, "min": 1, "max": 8}),
                "negative_instruction": ("STRING", {"default": DEFAULT_NEGATIVE_INSTRUCTION, "multiline": True}),
                "empty_instruction": ("STRING", {"default": "", "multiline": True}),
                "max_input_image_pixels": ("INT", {"default": 1024 * 1024, "min": 64 * 64, "max": 4096 * 4096}),
                "max_input_image_side_length": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                "max_vlm_input_pil_pixels": ("INT", {"default": 384 * 384, "min": 64 * 64, "max": 2048 * 2048}),
                "max_vlm_input_pil_side_length": ("INT", {"default": 768, "min": 64, "max": 4096}),
                "max_sequence_length": ("INT", {"default": 1024, "min": 32, "max": 8192}),
                "system_prompt_follows_task_type": ("BOOLEAN", {"default": True}),
                "use_boosted_orthogonal_guidance": ("BOOLEAN", {"default": False}),
                "bog_mu": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "bog_range_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bog_range_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bog_interval": ("INT", {"default": 3, "min": 1, "max": 50}),
                "use_dmd_student_inference": ("BOOLEAN", {"default": False}),
                "dmd_conditioning_sigma": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "instruction_override": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "BOOGU"

    def generate(
        self,
        pipeline,
        instruction: str,
        width: int,
        height: int,
        num_inference_steps: int,
        text_guidance_scale: float,
        image_guidance_scale: float,
        empty_instruction_guidance_scale: float,
        seed: int,
        num_images_per_instruction: int,
        negative_instruction: str,
        empty_instruction: str,
        max_input_image_pixels: int,
        max_input_image_side_length: int,
        max_vlm_input_pil_pixels: int,
        max_vlm_input_pil_side_length: int,
        max_sequence_length: int,
        system_prompt_follows_task_type: bool,
        use_boosted_orthogonal_guidance: bool,
        bog_mu: float,
        bog_range_start: float,
        bog_range_end: float,
        bog_interval: int,
        use_dmd_student_inference: bool,
        dmd_conditioning_sigma: float,
        input_image: Optional[torch.Tensor] = None,
        instruction_override: Optional[str] = None,
    ):
        if instruction_override is not None and isinstance(instruction_override, str) and instruction_override.strip():
            instruction = instruction_override.strip()

        input_images = None
        if input_image is not None:
            input_images = [_pil_from_image_tensor(input_image)]

        is_turbo = getattr(pipeline, "_boogu_kind", "") == "turbo"
        if is_turbo and input_image is not None:
            raise ValueError("Boogu Turbo supports pure text-to-image only. Disconnect input_image.")

        device = getattr(pipeline, "_boogu_device", "cuda:0")
        generator_device = device if str(device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        turbo_kwargs = {}
        if is_turbo:
            turbo_kwargs = {
                "use_dmd_student_inference": bool(use_dmd_student_inference),
                "dmd_conditioning_sigma": float(dmd_conditioning_sigma),
            }

        result = pipeline(
            instruction=instruction,
            input_images=input_images,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            text_guidance_scale=float(text_guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            empty_instruction_guidance_scale=float(empty_instruction_guidance_scale),
            negative_instruction=negative_instruction,
            empty_instruction=empty_instruction,
            num_images_per_instruction=int(num_images_per_instruction),
            generator=generator,
            max_input_image_pixels=int(max_input_image_pixels),
            max_input_image_side_length=int(max_input_image_side_length),
            max_vlm_input_pil_pixels=int(max_vlm_input_pil_pixels),
            max_vlm_input_pil_side_length=int(max_vlm_input_pil_side_length),
            max_sequence_length=int(max_sequence_length),
            truncate_instruction_sequence=False,
            system_prompt_follows_task_type=bool(system_prompt_follows_task_type),
            use_boosted_orthogonal_guidance=bool(use_boosted_orthogonal_guidance),
            bog_mu=float(bog_mu),
            bog_range=(float(bog_range_start), float(bog_range_end)),
            bog_interval=int(bog_interval),
            align_res=input_image is not None,
            output_type="pil",
            device=device,
            **turbo_kwargs,
        )
        return (_image_tensor_from_pil(list(result.images)),)


NODE_CLASS_MAPPINGS = {
    "BOOGULoadBasePipeline": BOOGULoadBasePipeline,
    "BOOGULoadEditPipeline": BOOGULoadEditPipeline,
    "BOOGULoadTurboPipeline": BOOGULoadTurboPipeline,
    "BOOGUGenerate": BOOGUGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BOOGULoadBasePipeline": "BOOGU: Load Base Pipeline",
    "BOOGULoadEditPipeline": "BOOGU: Load Edit Pipeline",
    "BOOGULoadTurboPipeline": "BOOGU: Load Turbo Pipeline",
    "BOOGUGenerate": "BOOGU: Generate",
}
