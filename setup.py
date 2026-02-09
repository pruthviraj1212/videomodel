from setuptools import setup, find_packages

setup(
    name="trash_detector",
    version="0.1.0",
    description="A YOLOv8-based trash detection package",
    author="Pruthviraj",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "opencv-python",
        "torch",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "trash-detector=trash_detector.cli:main",
        ],
    },
)
