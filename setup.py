from setuptools import setup, find_packages

setup(
    name="refat-unofficial",
    version="0.1.0",
    description="ReFAT: Refusal Feature Adversarial Training (Unofficial Implementation)",
    author="Anonymous",
    author_email="anonymous@example.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "numpy",
        "psutil",
        "peft",
        "jaxtyping",  # For type annotations
    ],
    python_requires=">=3.8",
    url="https://github.com/username/refat-unofficial",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 