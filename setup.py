from setuptools import setup, find_packages

setup(
    name="core_module",                 # Package name
    version="0.1.0",                    # Version
    author="Your Name",                 # Author information
    description="A reusable core module for pipelines",  # Short description
    long_description=open("README.md").read(),  # Optional long description
    long_description_content_type="text/markdown",
    url="https://github.com/fclairec/core_module",  # Repository URL
    packages=find_packages(include=["core_module", "core_module.*"]),  # Discover submodules    classifiers=[
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Python version compatibility
)
