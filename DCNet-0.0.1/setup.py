import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name="DCNet",
        version="0.0.1",
        author="XinWang",
        author_email="xindd_2014_2014@163.com",
        description="Using deep learning to unravel the cell profile from bulk expression data",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/xindd/DCNet",
        packages=setuptools.find_packages(),
        classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent",
        ],
    )