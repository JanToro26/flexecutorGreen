from setuptools import setup, find_packages
from itertools import chain

install_requires = [
    #"cloud-dataplug @ git+https://github.com/CLOUDLAB-URV/dataplug",
    "lithops",
    "boto3",
    "cloudpickle",
    "overrides",
    "deap",
    "scipy",
    "numpy",
    "pandas",
    "black",
]

extras_require = {
    "examples": [
        "S3path",
        "scikit-learn",
    ],
    "drawing": [
        "matplotlib",
        "networkx",
    ],
    "kubernetes": [
        "lithops[kubernetes]",
    ],
}

extras_require["all"] = list(set(chain.from_iterable(extras_require.values())))

setup(
    name="flexecutor",
    version="0.1.0",
    author="Daniel Barcelona, Ayman Bourramouss, Enrique Molina, Stepan Klymonchuk (DAGium donation)",
    author_email="daniel.barcelona@urv.cat, ayman.bourramouss@urv.cat, enrique.molina@urv.cat, "
    "stepan.klymonchuk@urv.cat",
    description="A flexible and DAG-optimized executor over Lithops",
    url="https://github.com/CLOUDLAB-URV/flexecutor",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    package_dir={"": "."},
)
