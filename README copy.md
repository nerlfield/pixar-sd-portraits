# pixar-sd-portraits

This repository is dedicated to transforming user-submitted portrait images into personalized cartoon characters that mimic the Pixar animation style using Stable Diffusion 1.5 or XL models.

## Objective

The primary objective of this project is to develop a minimal, workable code solution capable of:
1. Taking user-submitted portrait images
2. Transforming them into personalized cartoon characters in the Pixar animation style, preserving key facial attributes and a recognizable degree of the individual's identity

![Example Image](assets/result.jpg)

## Repository Contents

The repository features the following main components:
- `notebooks/`: notebooks that contain same code as in serve, but in decomposed easy to run way code.
- `src/`: I planned to store here some abstraction around pipeline in more optimized way, but now there is only utils file
- `assets/`: sample assets
- `serve/`: folder with two files:
   - '00_sdxl_pixar_lora_demo.py' - to run SDXL demo with face IP-Adapter + Pixar LoRA that I took from civitAI, you can find my fork here: https://huggingface.co/animte/pixar-sdxl-lora
   - '01_sd1.5_pixar_demo.py' - same as 00, but I use SD1.5 finetuned checkpoint on pixar images. It's lools much better than SDXL variant. I recommend run this demo.

## Environment Setup Instructions

To begin experimenting with Pixar-style portrait generation, set up your environment by following these steps:

1. Create a new conda environment:
   ```bash
   conda create --name pixarsd python=3.10
   ```

2. Activate the environment:
   ```bash
   conda activate pixarsd
   ```

3. Install system dependencies:
   ```bash
   sudo apt-get install cmake protobuf-compiler libprotobuf-dev
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the gradio demo:
   ```bash
   python serve/01_sd1.5_pixar_demo.py
   ```

6. Or you can run jupyter notebook and run notebooks in `notebooks` folder.
   ```bash
   jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root --notebook-dir=.
   ```

# Approach

![alt text](./examples/image.png)

Approach is pretty classic on this stage. I took IP-Adapter, SD model and Pixar LoRA (Low-Rank Adaptation).

In both pipelines I first take user picture, retrieve face region using insightface `buffalo_l` model, than I crop face and pass it as an condition for ip-adapter face sd model in image-2-image style. 

## Next steps for improvements:

This is POC right now, so it has a lot of directions to improve:
1. Generation speed: I've tested both pipelines on nvidia a100 videocard. It took aroun 10 seconds to generate 4 images, 1024,1024 resolution. There are modern ways to compile torch models, using for example https://github.com/chengzeyi/stable-fast or https://github.com/NVIDIA/TensorRT, these frameworks could speedup diffusion models up to 2 times! Also activation caching could help.
2. To improve results, it makes sense to collect dataset and finetune model in image-2-image setting. For example, we could generate more pairs (real photo - styled in pixar) and train more narrow, smaller, quantized model to generate such images.
3. IP-Adapter is only one approach from many. There are also ConsistentID (https://github.com/JackAILab/ConsistentID), InstantID  (https://instantid.github.io/), Photomaker (https://github.com/TencentARC/PhotoMaker), FastComposer (https://github.com/mit-han-lab/fastcomposer), FaceAdapter (https://huggingface.co/h94/IP-Adapter-FaceID), and many other. It make sense to experiment with each, they could return more consistent images with stable identity preserving due training on larger set with same usecase as we (avatar creation)

## Results:

### SD1.5:

![alt text](./examples/image-4.png)
![alt text](./examples/image-5.png)
![alt text](./examples/image-6.png)

### SDXL:

![alt text](./examples/image-8.png)
![alt text](./examples/image-2.png)
![alt text](./examples/image-7.png)

