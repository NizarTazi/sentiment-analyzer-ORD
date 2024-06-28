# Sentiment Analyzer API

This project provides a Flask-based API for predicting whether the sentiment of movie reviews is positive or negative using a pretrained GPT model and a Logistic Regression.

You can run this project by either creating a docker image in Docker, or by using Python in a local environment. 

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Option 1: Using Docker](#docker)
- [Option 2: Using Conda or Python environment](#conda)
- [Running the Application](#running-the-application)
- [Usage](#usage)

## Requirements
- [Docker](https://www.docker.com/products/docker-desktop)
- Python 3.8.5
- Conda or miniconda (optional)
  
## Setup

### Clone the Repository
First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/sentiment-analyzer-ORD.git
cd sentiment-analyzer-ORD
```


## Option 1: Using Docker
#### 1. Ensure Docker is running.
#### 2. Open Terminal: Open Git Bash, Command Prompt, or PowerShell.
#### 3. Navigate to Project Directory
   

### 4. Build the Docker image
To build the Docker image, run the following command in the project directory:
```sh
docker build -t sentiment-analyzer-api .
```

### 5. Run the Docker container
To run the Docker container, use the following command:
```sh
docker run -p 5000:5000 sentiment-analyzer-api
```


## Option 2: Using Conda or Python environment
### 1. Create a new Python environment (Optional)
It is recommended to create a new Conda environment. Please note that the used Python version is 3.8.5. Some packages could be deprecated in newer versions of Python.

```sh
conda create --name sent-analyzer-env python=3.8.5
conda activate sent-analyzer-env
```
### 2. Install requirements
Install the required dependencies in the new environment:
```sh
python -m pip install -r requirements.txt
```
### 3. Run the application
Run the application using the following command:
```sh
python sentiment_analyzer.py
```

## Running the application
### Browse to the Application

Open your web browser and navigate to: http://127.0.0.1:5000/

## Usage

    You will be prompted with a text box and a predict button.
    Type movie reviews into the text box and click the predict button.
    The algorithm will predict if the review is positive or negative.


