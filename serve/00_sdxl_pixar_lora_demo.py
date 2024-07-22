import gradio as gr
import torch
import diffusers
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis
from PIL import Image
import math
import numpy as np

negative_prompt = "noisy, messy, grainy, highly detailed, ultra textured, photo, NSFW"

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16
)

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, image_encoder=image_encoder,
).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.bin")

pipeline.scheduler = diffusers.EulerDiscreteScheduler.from_config(
    pipeline.scheduler.config
)

pipeline.load_lora_weights('animte/pixar-sdxl-lora', weight_name='PixarXL.safetensors', adapter_name="pixar-style")

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def generate_pixar_portrait(input_image, prompt="breathtaking 3D image in the pixar style, disney, cartoon, 4k,unreal engine, blender 8k,outdoor,natural lighting,adult,clear face,stock image", guidance_scale=6, ip_adapter_scale=0.5, lora_scale=0.8, num_images=2, strength=0.6, num_inference_steps=50):
    init_image = input_image.convert("RGB")
    
    pipeline.set_ip_adapter_scale(ip_adapter_scale)
    pipeline.set_adapters(["pixar-style"], adapter_weights=[lora_scale])
    
    faces = app.get(np.array(init_image))
    bbox = faces[0]['bbox']
    cropped_image = np.array(init_image)[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ip_adapter_image=cropped_image,
        num_images_per_prompt=num_images,
        target_size=(1024, 1024),
        clip_skip=2,
        high_noise_frac=0.8
    ).images
    
    return images

def main():
    with gr.Blocks() as iface:
        gr.Markdown("# Pixar-Style Portrait Generator")
        gr.Markdown("Upload a portrait image to transform it into a Pixar-style cartoon character.")
        
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload your portrait")
            output_gallery = gr.Gallery(label="Generated Pixar-style Portraits")
        
        with gr.Row():
            generate_button = gr.Button("Generate")
        
        with gr.Accordion("Settings", open=True):
            num_images = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Number of Images")
            guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=6, label="Guidance Scale")
            
        with gr.Accordion("Advanced Settings", open=False):
            ip_adapter_scale = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="IP-Adapter Scale")
            lora_scale = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.8, label="LoRA Scale")
            prompt = gr.Textbox(value="breathtaking 3D image in the pixar style, disney, cartoon, 4k,unreal engine, blender 8k,outdoor,natural lighting,adult,clear face,stock image", label="Prompt")
            strength = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.6, label="Strength")
            num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Number of Inference Steps")
        
        generate_button.click(
            generate_pixar_portrait,
            inputs=[input_image, prompt, guidance_scale, ip_adapter_scale, lora_scale, num_images, strength, num_inference_steps],
            outputs=output_gallery
        )

        gr.Examples(
            examples=[
                ["./assets/examples/andrej_karpathy.webp", "breathtaking 3D image in the pixar style, disney, cartoon, 4k,unreal engine, blender 8k,outdoor,natural lighting,adult,clear face,stock image", 6, 0.5, 0.8, 2, 0.6, 50],
                ["./assets/examples/andrew_ng.jpg", "breathtaking 3D image in the pixar style, disney, cartoon, 4k,unreal engine, blender 8k,outdoor,natural lighting,adult,clear face,stock image", 6, 0.5, 0.8, 2, 0.6, 50],
                ["./assets/examples/yann-lecun.jpg", "breathtaking 3D image in the pixar style, disney, cartoon, 4k,unreal engine, blender 8k,outdoor,natural lighting,adult,clear face,stock image", 6, 0.5, 0.8, 2, 0.6, 50]
            ],
            inputs=[input_image, prompt, guidance_scale, ip_adapter_scale, lora_scale, num_images, strength, num_inference_steps],
            fn=generate_pixar_portrait,
            cache_examples=True,
            outputs=output_gallery
        )

    iface.launch()

if __name__ == "__main__":
    main()