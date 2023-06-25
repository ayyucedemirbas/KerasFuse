<p align="center">
  <a href="https://github.com/ayyucedemirbas/KerasFuse"><img src="https://github.com/ayyucedemirbas/KerasFuse/assets/8023150/41d8880d-8117-448b-a725-2b72d2d08beb" alt="KerasFuse"></a>
</p>

<h1 align="center">KerasFuse</h1>


<p align="center">
  <img alt="GitHub" src="https://img.shields.io/github/license/ayyucedemirbas/Kerasfuse">
  <img alt="Tensorflow" src="https://img.shields.io/badge/Tensorflow-v2.12.0-%23FF6F00.svg?logo=Tensorflow&logoColor=white"/>
  <img alt="Keras" src="https://img.shields.io/badge/Keras-v2.12.0-%23D00000.svg?logo=Keras&logoColor=white"/>
  <img alt="Black" src="https://img.shields.io/badge/code%20style-black-black"/>
  <img alt="isort" src="https://img.shields.io/badge/isort-checked-yellow"/>
</p>
<p align="center">
<a href="https://pypi.org/project/kerasfuse" target="_blank">
    <img src="https://img.shields.io/pypi/v/kerasfuse?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/kerasfuse" target="_blank">
    <img src="https://img.shields.io/pypi/dm/kerasfuse?color=red" alt="Download Count">
</a>
<a href="https://pypi.org/project/kerasfuse" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/kerasfuse.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://pypi.org/project/kerasfuse" target="_blank">
    <img src="https://img.shields.io/pypi/status/kerasfuse?color=orange" alt="Project Status">
</a>
</p>

<h4 align="center">🚧 Warning this project is under heavy development and not ready for production. ABI changes can happen frequently until reach stable version 🚧 </h4>


KerasFuse is a Python library that combines the power of TensorFlow and Keras with various computer vision techniques for medical image analysis tasks. It provides a collection of modules and functions to facilitate the development of deep learning models in TensorFlow Keras for tasks such as image segmentation, classification, and more.



## Getting Started

## Requirements

KerasFuse is a project that relies heavily on the Tensorflow and Keras libraries. It is designed to work seamlessly with these powerful tools for deep learning and neural network development. In order to use KerasFuse effectively, please ensure that you have the following:

* Python 3.8+
* Tensorflow 2.12.0+
* Keras 2.12.0+
* OpenCV 4.7+
* Scikit-Learn 1.2.2+

## Installation

```console
$ pip install kerasfuse
---> 100%
```

## Development

#### Poetry Installation

```bash
poetry install
poetry shell
```

#### Tip

If you have multiple Python versions on your system, you can set your Python version by using `poetry env` . Here's an example of how to use it:

```bash
poetry env use python3.10
```

More details at
[poetry-switching-between-environments](https://python-poetry.org/docs/managing-environments/#switching-between-environments)

#### Pip Installations

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## License

This project is licensed under the terms of the GPL-3.0 license.
