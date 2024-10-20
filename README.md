# captylize
Easily extendable API for prototyping with ML models to analyze and generate content from images.

## Introduction

**Note: This project is under active development. Expect breaking changes. Use at your own risk.**

Captylize is a simple API designed to facilitate easy prototyping of Hugging Face models and other image analysis models. It provides a straightforward interface for analyzing images with the goal of:

- Analyzing images (Using classification for e.g. age, emotion, nsfw)
- Generating captions for images (for tagging datasets or building prompts)
- Detecting objects and faces in images (coming soon™)

## Features

- Image captioning using VIT-GPT2 and Florence-2 models
- Age estimation
- Emotion detection
- NSFW content detection
- Support for both image URL and file upload inputs
- Easy-to-use REST API endpoints

For information about the models used, see the [MODELS.md](MODELS.md) file.

## Todo

- [x] Implement basic image captioning with VIT-GPT2 model
- [x] Add support for advanced captioning with Florence-2 model
- [x] Implement age estimation endpoint
- [x] Implement emotion detection endpoint
- [x] Implement NSFW content detection endpoint
- [x] Support both image URL and file upload inputs
- [x] Set up FastAPI framework with proper routing and dependency injection
- [x] Implement model manager for easy model loading and unloading
- [x] Create basic error handling and input validation
- [ ] Add /detection endpoint for object & face detection
- [ ] Remove confusion about the 'default models' and centralize configuration of these so it propagates to the /docs page
- [ ] Add a /models endpoint that returns all models, loaded models, (tasks, general config?)
- [ ] Add a /health endpoint to gain an overview of the service and resource usage 
- [ ] Add support for batched inputs
- [ ] Add a proper Prediction response object (metrics, batched predictions- what else?)
- [ ] Figure out an easier way to add new models
- [ ] Docker support 


- [x] Don't forget to have fun!




## Installation

**It is recommended to use a virtual environment to install the project dependencies.**

First create and activate a virtual environment with python 3.11 or later.

Then install PyTorch 2 or later (project is developed using PyTorch 2.4.1)

Use the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the CPU or GPU version.

Then install the project dependencies:

**With pip:**

```bash
pip install -r requirements.txt
```

**With poetry:**

```bash
poetry install
```


## Usage

To run the API locally (in development mode)

Execute the `run_dev.sh` script.

Or the command:

```bash
uvicorn captylize.main:app --reload
```




Basic usage examples:

1. Image Captioning:
   ```bash
   POST /api/v1/generations/captions/vit
   POST /api/v1/generations/captions/florence-2
   ```

2. Age Estimation:
   ```bash
   POST /api/v1/analyses/ages
   ```

3. Emotion Detection:
   ```bash
   POST /api/v1/analyses/emotions
   ```

4. NSFW Detection:
   ```bash
   POST /api/v1/analyses/nsfw
   ```

5. Object Detection (coming soon™):
   ```bash
   POST /api/v1/detections/objects
   ```

6. Face Detection (coming soon™):
   ```bash
   POST /api/v1/detections/faces
   ```

For detailed API documentation, run the server and visit `/docs` or `/redoc`.


## License

This project and code within is licensed under the MIT License. Models referenced in this project are licensed under their respective licenses.