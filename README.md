# Ping Pong Ball Tracker

This application uses computer vision to track a ping pong ball and move a stick figure to catch it. It also predicts the ball's trajectory to determine where it will intersect with the stick figure's paddle.

## Features

- Real-time ping pong ball tracking using color detection
- Stick figure that moves to catch the ball
- Trajectory prediction to anticipate where the ball will land
- Simple scoring system

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- SciPy (for curve fitting)

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main application:
   ```
   python main.py
   ```

2. Hold a ping pong ball (preferably orange) in front of your webcam.
3. The stick figure will try to catch the ball based on its predicted trajectory.
4. Press ESC to quit the application.

## Customization

- Adjust the ball color range in `BallTracker` class if your ping pong ball has a different color.
- Modify the stick figure size and appearance in the `StickFigure` class.

## Troubleshooting

- If the ball is not being detected, try adjusting the color thresholds in the `BallTracker` class.
- Make sure there is adequate lighting for the webcam to detect the ball.
- If the trajectory prediction is inaccurate, try to move the ball more smoothly.
