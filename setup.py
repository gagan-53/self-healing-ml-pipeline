from setuptools import setup, find_packages
setup(
    name="sh_mlp",
    version="0.1.0",
    description="Self-Healing ML Pipeline — Phase 3 Prototype",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "psutil>=5.8",
    ],
)
