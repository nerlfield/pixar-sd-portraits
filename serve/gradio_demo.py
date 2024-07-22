import gradio as gr
import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
import lovely_tensors as lt

lt.monkey_patch()

negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

def generate_pixar_portrait(input_image, num_images, guidance_scale, ip_adapter_scale, pixar_style_weight, modern_pixar_weight, pipeline):
    ip_image = input_image.convert("RGB")
    
    pipeline.set_ip_adapter_scale(ip_adapter_scale)
    pipeline.set_adapters(["pixar-style", "modern-pixar"], adapter_weights=[pixar_style_weight, modern_pixar_weight])
    
    generator = torch.Generator(device="cpu").manual_seed(4)
    images = pipeline(
        prompt="cartoon, portrait in modern-pixar style, best quality, high quality, pixar-style",
        negative_prompt=negative_prompt,
        ip_adapter_image=ip_image,
        generator=generator,
        num_images_per_prompt=num_images,
        guidance_scale=guidance_scale
    ).images
    
    return images

def main():
    # Load models and pipelines
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )

    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

    # Load LoRAs
    pipeline.load_lora_weights('ntc-ai/SDXL-LoRA-slider.pixar-style', weight_name='pixar-style.safetensors', adapter_name="pixar-style")
    pipeline.load_lora_weights('jbilcke-hf/sdxl-modern-pixar', weight_name = "pytorch_lora_weights.safetensors", adapter_name="modern-pixar")

    # Gradio interface
    with gr.Blocks() as iface:
        gr.Markdown("# Pixar-Style Portrait Generator")
        gr.Markdown("Upload a portrait image to transform it into a Pixar-style cartoon character.")
        
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload your portrait")
            output_gallery = gr.Gallery(label="Generated Pixar-style Portraits")
        
        with gr.Row():
            generate_button = gr.Button("Generate")
        
        with gr.Accordion("Basic Settings", open=True):
            num_images = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Number of Images")
            guidance_scale = gr.Slider(minimum=1, maximum=10, step=0.5, value=6, label="Guidance Scale")
        
        with gr.Accordion("Advanced Settings", open=False):
            ip_adapter_scale = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="IP-Adapter Scale")
            pixar_style_weight = gr.Slider(minimum=0, maximum=1, step=0.05, value=1.0, label="Pixar-Style LoRA Weight")
            modern_pixar_weight = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.7, label="Modern-Pixar LoRA Weight")
        
        generate_button.click(
            generate_pixar_portrait,
            inputs=[input_image, num_images, guidance_scale, ip_adapter_scale, pixar_style_weight, modern_pixar_weight],
            outputs=output_gallery,
            _js="(input_image, num_images, guidance_scale, ip_adapter_scale, pixar_style_weight, modern_pixar_weight) => { return [input_image, num_images, guidance_scale, ip_adapter_scale, pixar_style_weight, modern_pixar_weight, pipeline]; }"
        )

    iface.launch()

if __name__ == "__main__":
    main()