# Trainify! AI-Powered Fitness Trainer

This repository contains the code for an AI-powered virtual fitness trainer. The project uses **MediaPipe** and **OpenCV** for pose tracking, alongside an **LSTM neural network** model to smooth motion data and predict future joint angles, enhancing the accuracy of exercise monitoring. The trainer currently supports exercises such as jumping jacks, bicep curls, shoulder presses, squats, and push-ups.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project employs a **Long Short-Term Memory (LSTM)** model to smooth and predict joint angles in real-time video feed for better tracking and feedback. Using **MediaPipe's Pose Detection** with OpenCV, joint angles for selected exercises are extracted from video frames and fed into an LSTM for movement trend analysis and prediction.

The goal is to provide accurate feedback by reducing jitter and noise in joint tracking data. The system can help track a variety of exercises and guide users toward proper form.

This project was made by Piyush Prakash, Chetna Rajeev, Lathika Kommineni and Harshini Kasturi

## Features

- **Real-Time Pose Tracking**: Uses MediaPipe's Pose detection to track major joints.
- **Joint Angle Calculation**: Calculates specific joint angles needed for each exercise.
- **LSTM Model for Motion Smoothing and Prediction**: Reduces jitter and predicts joint movement trends.
- **Supports Multiple Exercises**: Tracks jumping jacks, bicep curls, shoulder press, squats, and push-ups.

## Installation

### Prerequisites
1. **Python 3.8+**
2. **MediaPipe** and **OpenCV**

Clone this repository:
```bash
git clone https://github.com/Piyush240604/Trainify-A-Virtual-AI-Trainer
cd Trainify-A-Virtual-AI-Trainer
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. **Collect Videos**: Record or collect sample videos of each exercise with multiple repetitions.
2. **Extract Joint Angles**: Use `LSTM_train.ipynb` to extract joint angles for each exercise video.
    - The `LSTM_train.py` script processes each frame and calculates relevant joint angles.

3. **Preprocess Data**: Normalize angles and prepare sequences for LSTM input in `LSTM_train.ipynb`.

## Usage

1. **Pose Tracking and Angle Calculation**
   - Run `LSTM_train.ipynb` to analyze a video and output joint angles for each frame.


2. **Training the LSTM Model**
   - Use `LSTM_train.ipynb` to train an LSTM model on the joint angle data.

   This script will save the trained model to `models/`.

3. **Real-Time Exercise Monitoring**
   - Run the main program, `FitnessTrainer_integration.py`, to track exercises in real-time, with predictions and smoothing applied.


   - **Note**: Choose an exercise at the start to allow the program to use the correct model for smoothing and predictions.

## Model Training

The LSTM model is trained individually for each exercise to optimize performance and maintain efficiency. Training involves:
- **Hyperparameters**: Customize for each exercise, such as sequence length, hidden units, learning rate, and epochs.
- **Input Data**: Joint angles calculated from frames, organized as sequential time-series data.
  
Each exercise has its own trained LSTM model, which takes joint angle sequences as input and outputs smoothed/predicted angles.

### Training Example

The `LSTM_train.ipynb` file has a configuration for jumping jacks:

```python
sequence_length = 20
hidden_size = 50
learning_rate = 0.001
epochs = 50
```

To train the model for other exercises, adjust `--exercise` to specify the correct dataset and parameters.

## Demo

To view a demo of the Fitness Trainer in action, run:

## Future Work

- **Add More Exercises**: Extend support to more exercises with targeted joints.
- **Real-Time Feedback**: Provide more detailed feedback for form correction.
- **Multi-Exercise LSTM Model**: Explore a universal LSTM model for multiple exercises.
- **Proper GUI**: Implement a better GUI for intuitive user interface

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repo and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
