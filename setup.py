import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="stockDL",
    version="0.2.1",
    description="Predicts the Gross Yield, Annual Yield and Net Yield of a user given stock ticker.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ashishpapanai/stockDL",
    author="Ashish Papanai",
    author_email="ashishpapanai00@gmail.com",
    py_modules=["calculations", "data", "main", "market", "models", "plots", "preprocessing", "results", "train"],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["stockDL"],
    include_package_data=True,
    install_requires=["pandas", "numpy", "matplotlib", "keras", "tensorflow", "yahoo-finance", "yfinance"],
    entry_points={
        "console_scripts": [
            "stocksDL=stocksDL.__main__:main",
        ]
    },
)
