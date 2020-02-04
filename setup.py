from setuptools import setup
from setuptools import find_packages
from os.path import join, dirname

with open(join(dirname(__file__), "eagerpy/VERSION")) as f:
    version = f.read().strip()

try:
    # obtain long description from README
    readme_path = join(dirname(__file__), "README.rst")
    with open(readme_path, encoding="utf-8") as f:
        README = f.read()
except IOError:
    README = ""


install_requires = ["numpy", "typing-extensions"]
tests_require = ["pytest", "pytest-cov"]

setup(
    name="eagerpy",
    version=version,
    description="EagerPy is a thin wrapper around PyTorch, TensorFlow Eager, JAX and NumPy that unifies their interface and thus allows writing code that works natively across all of them.",
    long_description=README,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    author="Jonas Rauber",
    author_email="jonas.rauber@bethgelab.org",
    url="https://github.com/jonasrauber/eagerpy",
    license="",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={"testing": tests_require},
)
