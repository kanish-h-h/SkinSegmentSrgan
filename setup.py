from setuptools import setup, find_packages

setup(
    name="skinsegmentsrgan",
    version="0.1.0",
    description="Skin Disease Segmentation + SRGAN for Low-Contrast Images",
    author="Kanish",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.10.0",
        "opencv-python>=4.7.0",
        "scikit-image>=0.20.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "skinsegmentsrgan=inference.cli:main",
        ],
    },
    python_requires=">=3.8",
)
