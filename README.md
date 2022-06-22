# watermark-detection
A repository containing datasets and tools to train a watermark classifier.

## Datasets
The datasets folder contains WebDataset files using the following keys:
- key: A unique identifier within the dataset
- url: The URL of the image data
- caption: The caption describing the images

## Dataset annotation
Preliminary dataset annotations and WIP tools can be found on https://github.com/robvanvolt/DALLE-tools.
More mature tools and completely annotated datasets will be transferred to this repository.
Feel free to adjust the structure, upload new annotations or annotation tools.

## Training
The training and evaluation source code is availible under `./training/`
Note: Deepspeed is still unstable

## Models
The current model is at `./models/watermark_model_v1.pt` and availible in the release as well

## Usage
See `example_use.py`

## Web Annotator
WIP - A tool to annotate the url-caption datasets online will be shortly uploaded.
