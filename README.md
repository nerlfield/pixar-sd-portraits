# pixar-sd-portraits

This repository is dedicated to transforming user-submitted portrait images into personalized cartoon characters that mimic the Pixar animation style using Stable Diffusion technology.

## Objective

The primary objective of this project is to develop a minimal, workable code solution capable of:
1. Taking user-submitted portrait images
2. Transforming them into personalized cartoon characters in the Pixar animation style
3. Preserving key facial attributes and a recognizable degree of the individual's identity
4. Applying the distinctive, animated appeal seen in Pixar films

<!-- ![Example Image](assets/example.png) -->

## Repository Contents

The repository features the following main components:
- `notebooks/`: Jupyter notebooks for experimentation and demonstration
- `src/`: Source code for the image transformation pipeline
- `assets/`: Sample images and resources
- `requirements.txt`: List of Python dependencies

## Environment Setup Instructions

To begin experimenting with Pixar-style portrait generation, set up your environment by following these steps:

1. Create a new conda environment:
   ```bash
   conda create --name pixarsd2 python=3.10
   ```

2. Activate the environment:
   ```bash
   conda activate pixarsd2
   ```

3. Install system dependencies:
   pip install .


   ```bash
   sudo apt-get install cmake protobuf-compiler libprotobuf-dev
   ```

4. Install Python dependencies:
   ```bash
   pip install pybind11
   pip install onnx==1.11.0 --find-links https://download.pytorch.org/whl/onnx_stable.html
   pip install -r requirements.txt
   pip install insightface
   ```

5. Start Jupyter Lab to access the notebooks:
   ```bash
   jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root --notebook-dir=.
   ```