# Facial Feature Detection

## Overview
##### `notebooks/` - showcases the process of the ML pipeline
- `1. extract_faces.ipynb` - data cleaning
- `2. pytorch.ipynb` - training, validation
- `3. eval.ipynb` - testing, visualization of results

##### `main.py` - entrypoint for running the live facial feature detection

##### `webpage.py` - entrypoint for running the flask hosted webpage

## Instructions
To run `main.py` or `webpage.py`, you need the model files. You can either:
1) download the pretrained models here:
   - [cae.pth (optional)](https://drive.google.com/file/d/1QNDieM7NEUqWpwn3DtOpAfKPJ2XONOfe/view?usp=sharing)
   - [cls.pth (required)](https://drive.google.com/file/d/1XzbUSz6HSrlffBfI3QwoIxa8i1yOpHGi/view?usp=sharing)

OR

2) download the [CelebA dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) and train your own models by running the `notebooks` in order