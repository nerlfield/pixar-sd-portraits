import gradio as gr
from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
import diffusers
import torch
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import math
import numpy as np
from insightface.app import FaceAnalysis

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stablediffusionapi/disney-pixar-cartoon", torch_dtype=torch.float16
).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")

pipeline.scheduler = diffusers.EulerDiscreteScheduler.from_config(
    pipeline.scheduler.config
)

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

negative_prompt = "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo, NSFW"

def create_collage(images, size=(1024, 1024)):
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    img_width = size[0] // cols
    img_height = size[1] // rows

    collage = Image.new('RGB', size)

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        # Resize and crop the image to fit
        img_aspect = img.width / img.height
        cell_aspect = img_width / img_height
        if img_aspect > cell_aspect:
            new_width = int(img_height * img_aspect)
            img = img.resize((new_width, img_height), Image.LANCZOS)
            left = (new_width - img_width) // 2
            img = img.crop((left, 0, left + img_width, img_height))
        else:
            new_height = int(img_width / img_aspect)
            img = img.resize((img_width, new_height), Image.LANCZOS)
            top = (new_height - img_height) // 2
            img = img.crop((0, top, img_width, top + img_height))

        collage.paste(img, (x_offset, y_offset))
        x_offset += img_width
        if (i + 1) % cols == 0:
            x_offset = 0
            y_offset += img_height

    return collage

def generate_pixar_portrait(input_image, prompt, guidance_scale, ip_adapter_scale, num_images, n_infer_steps, strength):
    init_image = input_image.convert("RGB")
    
    faces = app.get(np.array(init_image))
    bbox = faces[0]['bbox']
    cropped_image = np.array(init_image)[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    
    pipeline.set_ip_adapter_scale(ip_adapter_scale)
    
    images = pipeline(prompt, 
                      negative_prompt=negative_prompt, 
                      image=init_image, 
                      strength=strength, 
                      guidance_scale=guidance_scale,
                      num_images_per_prompt=num_images,
                      num_inference_steps=n_infer_steps,
                      target_size=(1024, 1024),
                      ip_adapter_image=cropped_image,
                      clip_skip=1,
                      high_noise_frac=0.8).images
    
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
            ip_adapter_scale = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.8, label="IP-Adapter Scale")
            num_images = gr.Slider(minimum=1, maximum=4, step=1, value=4, label="Number of Images")
            
        with gr.Accordion("Advanced Settings", open=False):
            prompt = gr.Textbox(value="breathtaking 3D image in the pixar style, disney, cartoon, 4k,unreal engine, blender 8k,outdoor,natural lighting,adult,clear face,stock image", label="Prompt")
            guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=10.5, label="Guidance Scale")
            n_infer_steps = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Number of Inference Steps")
            strength = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.6, label="Strength")
        
        generate_button.click(
            generate_pixar_portrait,
            inputs=[input_image, prompt, guidance_scale, ip_adapter_scale, num_images, n_infer_steps, strength],
            outputs=output_gallery
        )

        gr.Examples(
            examples=[
                "../assets/examples/example1.jpg",
                "../assets/examples/example2.jpg",
                "../assets/examples/example3.jpg"
            ],
            inputs=input_image
        )

    iface.launch()

if __name__ == "__main__":
    main()