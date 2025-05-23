import argparse
import gc
import os
import random
import time
from typing import Literal, Optional

import imageio
import torch
from diffusers.utils import load_image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop
from skyreels_v2_infer.pipelines import Text2VideoPipeline

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}


def create_video_from_prompt_or_image(
    save_dir="video_out",
    model_id="Skywork/SkyReels-V2-T2V-14B-540P",
    resolution="540P",
    num_frames=97,
    image_path=None,
    guidance_scale=6.0,
    shift=8.0,
    inference_steps=30,
    use_usp=False,
    offload=False,
    fps=24,
    seed=None,
    prompt="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
    prompt_enhancer=False,
    teacache=False,
    teacache_thresh=0.2,
    use_ret_steps=False,
):
    model_id = download_model(model_id)
    print("model_id:", model_id)

    if use_usp and seed is None:
        raise ValueError("USP mode requires a fixed seed")
    if seed is None:
        seed = int(random.randrange(4294967294))

    height, width = {
        "540P": (544, 960),
        "720P": (720, 1280),
    }.get(resolution, (None, None))
    if height is None:
        raise ValueError(f"Invalid resolution: {resolution}")

    image = load_image(image_path).convert("RGB") if image_path else None
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
        "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
        "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
        "background, three legs, many people in the background, walking backwards"
    )

    local_rank = 0
    if use_usp:
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        init_distributed_environment(rank=local_rank, world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )

    if prompt_enhancer and image is None:
        print(f"init prompt enhancer")
        enhancer = PromptEnhancer()
        prompt = enhancer(prompt)
        print(f"enhanced prompt: {prompt}")
        del enhancer
        gc.collect()
        torch.cuda.empty_cache()

    if image is None:
        assert "T2V" in model_id, f"Expected T2V model for text input: {model_id}"
        print("init text2video pipeline")
        pipe = Text2VideoPipeline(model_path=model_id, dit_path=model_id, use_usp=use_usp, offload=offload)
    else:
        assert "I2V" in model_id, f"Expected I2V model for image input: {model_id}"
        print("init image2video pipeline")
        pipe = Image2VideoPipeline(model_path=model_id, dit_path=model_id, use_usp=use_usp, offload=offload)
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        image = resizecrop(image, height, width)

    if teacache:
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=inference_steps,
            teacache_thresh=teacache_thresh,
            use_ret_steps=use_ret_steps,
            ckpt_dir=model_id,
        )

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "num_inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "height": height,
        "width": width,
    }

    if image is not None:
        kwargs["image"] = image.convert("RGB")

    os.makedirs(save_dir, exist_ok=True)

    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        print(f"Running inference with kwargs: {kwargs}")
        video_frames = pipe(**kwargs)[0]

    output_path = None
    if local_rank == 0:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        filename = f"{prompt[:100].replace('/','')}_{seed}_{timestamp}.mp4"
        output_path = os.path.join(save_dir, filename)
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
        print(f"Video saved at: {output_path}")

    return output_path

def main():
    # Convert args to dictionary and call the main function
    create_video_from_prompt_or_image(
        model_id="Skywork/SkyReels-V2-I2V-1.3B-540P",
        prompt="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
        resolution="540P",
        num_frames=97,
        image_path="swan.png",
        guidance_scale=5.0,
        shift=3.0,
        fps=24,
        offload=True,
        teacache=True,
        use_ret_steps=True,
        teacache_thresh=0.3,
    )


if __name__ == "__main__":
    main()