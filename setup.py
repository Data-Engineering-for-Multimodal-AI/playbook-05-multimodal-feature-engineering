from setuptools import setup, find_packages

# Read requirements
# (skips the comment header that uv export writes)
with open('requirements.txt') as f:
    requirements = [line for line in f.read().splitlines() if line.strip() and not line.startswith('#')]

setup(
    name="waymo-multimodal-pipeline",
    version="0.1.0",
    description="Multimodal feature engineering pipeline for Waymo Motion and Reasoning datasets",
    author="Data Engineering for Multimodal AI",
    author_email="",
    url="https://github.com/Data-Engineering-for-Multimodal-AI/playbook-05-multimodal-feature-engineering",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
) 