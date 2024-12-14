import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from ddpm import DDPMsampler
from utilis import get_time_embedding, rescale


WIDTH = 512
HEIGHT = 512

LATENT_WIDTH = 512 // 8
LATENT_HEIGHT = 512 // 8


def generate(
    prompt: str,
    uncond_prompt: str,
    schedular="ddpm",
    strength=0.8,
    classifier_free_guidance=True,
    classifier_free_guidance_scale=7.5,
    inference_steps=50,
    input_image=None,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    Tokonizer=None,
    output="image",
):

    with torch.no_grad():
        if 1 < strength or strength < 0:
            raise ValueError("Strength value should be between 0 and 1")

        to_idle = lambda x: x
        if idle_device:
            to_idle = lambda x: x.to(idle_device)

        # random number generator
        generator = torch.Generator(device=device).seed()
        if seed:
            generator.manual_seed(seed)

        clip = models["CLIP"]
        clip.to(device)

        # 1st tokanize the prompt
        # adding padding to the prompt if it does not meet the max length of the sequence 77

        # (batch_size , seq_len)
        cond_tokens = Tokonizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        #  (batch_size, seq_len) -> (batch_size, seq_len, dim) (batch_size , 77 , 768)
        cond_context = clip(cond_tokens)
        context = cond_context
        if classifier_free_guidance:

            uncond_tokens = Tokonizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_prompt, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            context = torch.concat([cond_context, uncond_context], dim=0)

        del uncond_context, cond_context
        to_idle(clip)

        if schedular != "ddpm":
            raise ValueError(f"Unknown schedular {schedular}")

        schedular = DDPMsampler(generator)
        schedular.set_inference_timesteps(inference_steps)

        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            vae_encoder = models["encoder"]
            vae_encoder.to(device)

            input_image_tensor = input_image.resize(HEIGHT, WIDTH)
            # (height , width , channel)
            input_image_tensor = np.array(input_image_tensor)

            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )

            # adding batch size
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = rescale(input_image, (0, 255), (-1, 1), clamp=True)
            # (batch_size, H, W, C) -> (Batch_size, C, W, H)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            latents = vae_encoder(input_image_tensor)
            schedular.set_strength(strength)
            latents = schedular.add_noise(latents, schedular.timesteps[0])
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(schedular.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embeding = get_time_embedding(timestep).to(device)
            model_input = latents

            model_output = diffusion(model_input, context, time_embeding)

            if classifier_free_guidance:
                cond_output, uncond_output = model_output.chunk(2, dim=0)

                # the output is the noise
                model_output = (
                    classifier_free_guidance_scale(cond_output - uncond_output)
                    + uncond_output
                )
            # schedular removing the noise
            latents = schedular.step(
                timestep,
                latents,
                model_output,
            )

        to_idle(diffusion)

        if output == "latent":
            return latents[0]

        vae_decoder = models["decoder"]

        vae_decoder.to(device)

        image = rescale(latents, (-1, 1), (0, 255))
        to_idle(image)
        image = image.permute(0, 2, 3, 1).type(torch.uint8).numpy()

        return image[0]
