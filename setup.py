from setuptools import setup, find_packages

setup(
    name="pixar_sd_portraits",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "diffusers==0.29.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.9.1",
        "scikit-learn==1.5.1",
        "plotly==5.22.0",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        "scipy==1.14.0",
        "jupyterlab==4.2.4",
        "ipywidgets==8.1.3",
        "sql==2022.4.0",
        "lovely-tensors==0.1.16",
        "tqdm==4.66.4",
        "protobuf==5.27.2",
        "gradio==4.38.1",
    ],
    entry_points={
        "console_scripts": [
            "pixar-portraits=serve.gradio_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.rst"],
        "pixar_sd_portraits": ["*.msg"],
    },
)