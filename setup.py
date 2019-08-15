# Coding: utf-8
# Time  : 2019/8/15
# Author: Li Xiang
# @Email: 22967920@qq.com
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text_sum",
    version="0.0.1",
    author="Li Xiang",
    author_email="22967920@qq.com",
    description="Text summarization for news",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)