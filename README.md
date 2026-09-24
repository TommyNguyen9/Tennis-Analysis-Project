# Tennis Analysis Project

This is a computer vision based tennis analysis system using YOLOv8 and custom trained models to detect players, track the tennis ball, detect court keypoints, estimate shot speeds
and visualise player movements on a mini court.

## Overview

This project analyses a tennis match footage from a Vienna ATP tour using computer vision and ML techniques.

The system peforms:

- Player detection and tracking
- Tennis ball detection and tracking
- Ball position interpolation
- Ball hit detection
- Tennis court keypoint detection
- Mini court visualisation
- Ball & Player position mapping
- Shot speed estimatino
- Player movement speed estimates
- Match statistics overlay

## Technologies Used:

- Python
- YOLOv8
- OpenCV
- PyTorch
- Pandas
- Numpy
- Ultralytics


### Running the Project:

Clone the repository:

```
git clone https://github.com/TommyNguyen9/Tennis-Analysis-Project.git
cd Tennis-Analysis-Project
```

Install the required Python packages.

```
pip install ultralytics opencv-python torch pandas numpy
```
Place your tennis video inside:

```text
input_videos/
```

### Current Models

The main analysis pipeline currently uses:

- YOLOv8 for player detection
- Custom YOLOv8 tennis ball detector
- Custom-trained tennis court keypoint mode


