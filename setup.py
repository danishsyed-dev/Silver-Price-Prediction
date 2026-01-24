from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="SilverPricePrediction",
    version="1.0.0",
    author="Danish",
    author_email="your_email@example.com",
    description="End-to-end machine learning project for silver price prediction",
    long_description=open("README_silver.md").read() if __import__("os").path.exists("README_silver.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Silver-Price-Prediction",
    install_requires=get_requirements("requirements_silver.txt"),
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="silver price prediction machine-learning time-series forecasting",
    entry_points={
        "console_scripts": [
            "silver-train=src.SilverPricePrediction.pipelines.Training_pipeline:run_training_pipeline",
        ],
    },
)
