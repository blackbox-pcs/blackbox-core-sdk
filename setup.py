"""
Setup script for Black Box Precision Core SDK
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="blackboxpcs",
    version="1.0.0",
    author="The XAI Lab",
    description="Black Box Precision: Unlocking High-Stakes Performance with Explainable AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/blackboxpcs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "shap>=0.41.0",
        "lime>=0.2.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
    keywords="xai explainable-ai shap lime machine-learning ai interpretability",
    project_urls={
        "Documentation": "https://github.com/your-org/blackboxpcs",
        "Source": "https://github.com/your-org/blackboxpcs",
        "Tracker": "https://github.com/your-org/blackboxpcs/issues",
    },
)


