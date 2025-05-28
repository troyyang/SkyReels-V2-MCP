import os
import gc
import time
import random
import torch
import imageio
from diffusers.utils import load_image
from moviepy.editor import VideoFileClip

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop

def get_video_num_frames_moviepy(video_path):
    with VideoFileClip(video_path) as clip:
        num_frames = sum(1 for _ in clip.iter_frames())
        return clip.size, num_frames

def generate_video(
    save_dir="df",
    model_id="Skywork/SkyReels-V2-DF-1.3B-540P",
    resolution="960*544",
    num_frames=97,
    image=None,
    end_image=None,
    video_path='',
    ar_step=0,
    causal_attention=False,
    causal_block_size=1,
    base_num_frames=97,
    overlap_history=None,
    addnoise_condition=0,
    guidance_scale=6.0,
    shift=8.0,
    inference_steps=30,
    use_usp=False,
    offload=False,
    fps=24,
    seed=None,
    prompt=(
        "A woman in a leather jacket and sunglasses riding a vintage motorcycle through "
        "a desert highway at sunset, her hair blowing wildly in the wind as the motorcycle "
        "kicks up dust, with the golden sun casting long shadows across the barren landscape."
    ),
    prompt_enhancer=False,
    teacache=False,
    teacache_thresh=0.2,
    use_ret_steps=False
):
    # Download and prepare the model
    model_id = download_model(model_id)
    print("model_id:", model_id)

    # Seed initialization
    if use_usp and seed is None:
        raise ValueError("USP mode requires a fixed seed.")
    if seed is None:
        random.seed(time.time())
        seed = int(random.randrange(4294967294))

    # Set resolution
    width, height = {
        "544*960": (544, 960),
        "960*544": (960, 544),
        "720*1280": (720, 1280),
        "1280*720": (1280, 720),
    }.get(resolution, (None, None))

    if height is None:
        raise ValueError(f"Invalid resolution: {resolution}")

    # Validate parameters
    if num_frames > base_num_frames and overlap_history is None:
        raise ValueError(
            'For long video generation, "overlap_history" must be specified. Recommended values are 17 or 37.'
        )
    if addnoise_condition > 60:
        print(
            f'Warning: "addnoise_condition" is set to {addnoise_condition}, which may cause inconsistency in long video generation. Recommended value is 20.'
        )

    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    local_rank = 0

    # Initialize distributed environment if using USP
    if use_usp:
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = "cuda"

        init_distributed_environment(rank=local_rank, world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )

    # Enhance prompt if required
    prompt_input = prompt
    if prompt_enhancer and image is None:
        print("Initializing prompt enhancer...")
        enhancer = PromptEnhancer()
        prompt_input = enhancer(prompt_input)
        print(f"Enhanced prompt: {prompt_input}")
        del enhancer
        gc.collect()
        torch.cuda.empty_cache()

    # Initialize the pipeline
    pipe = DiffusionForcingPipeline(
        model_id,
        dit_path=model_id,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=use_usp,
        offload=offload,
    )

    # Configure causal attention if enabled
    if causal_attention:
        pipe.transformer.set_ar_attention(causal_block_size)

    # Initialize TEACache if enabled
    if teacache:
        if ar_step > 0:
            num_steps = inference_steps + (((base_num_frames - 1) // 4 + 1) // causal_block_size - 1) * ar_step
            print('num_steps:', num_steps)
        else:
            num_steps = inference_steps
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=num_steps,
            teacache_thresh=teacache_thresh,
            use_ret_steps=use_ret_steps,
            ckpt_dir=model_id
        )

    print(f"Prompt: {prompt_input}")
    print(f"Guidance Scale: {guidance_scale}")

    # Generate video
    if os.path.exists(video_path):
        (v_width, v_height), input_num_frames = get_video_num_frames_moviepy(video_path)
        if input_num_frames < overlap_history:
            raise ValueError("The input video is too short for the specified overlap history.")

        if v_height > v_width:
            width, height = height, width

        video_frames = pipe.extend_video(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            prefix_video_path=video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            overlap_history=overlap_history,
            addnoise_condition=addnoise_condition,
            base_num_frames=base_num_frames,
            ar_step=ar_step,
            causal_block_size=causal_block_size,
            fps=fps,
        )[0]
    else:
        # Load and preprocess images if provided
        if image:
            image = load_image(image)
            image_width, image_height = image.size
            if image_height > image_width:
                height, width = width, height
            image = resizecrop(image, height, width)
            if end_image:
                end_image = load_image(end_image)
                end_image = resizecrop(end_image, height, width)

        image_rgb = image.convert("RGB") if image else None
        end_image_rgb = end_image.convert("RGB") if end_image else None

        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                image=image_rgb,
                end_image=end_image_rgb,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                overlap_history=overlap_history,
                addnoise_condition=addnoise_condition,
                base_num_frames=base_num_frames,
                ar_step=ar_step,
                causal_block_size=causal_block_size,
                fps=fps,
            )[0]

    # Save the generated video
    output_path = None
    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{prompt[:100].replace('/','')}_{seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
        print(f"Video saved to {output_path}")

    return output_path

# 使用示例
if __name__ == "__main__":
    output_path = generate_video(
        outdir="df",
        model_id="Skywork/SkyReels-V2-DF-1.3B-540P",
        resolution="540P",
        # ar_step=0,
        ar_step=5,
        causal_block_size=5,
        base_num_frames=97,
        num_frames=377,
        overlap_history=17,
        prompt="A cat chases a butterfly in a sunlit garden with colorful flowers.",
        addnoise_condition=20,
        offload=True,
        # teacache=True,
        # use_ret_steps=True,
        # teacache_thresh=True
    )