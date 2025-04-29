from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="waymo-multimodal-pipeline",
    version="0.1.0",
    description="Multimodal feature engineering pipeline for Waymo Motion and Reasoning datasets",
    author="Data Engineering for Multimodal AI",
    author_email="",
    url="https://github.com/Data-Engineering-for-Multimodal-AI/pb5-mm-feature-engineering-pipeline",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 