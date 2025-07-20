import cv2
import numpy as np
import time
from utils.ball_tracker import BallTracker
from utils.trajectory import TrajectoryPredictor
from utils.stick_figure import StickFigure

def main():
    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return
    
    # Get screen dimensions
    success, frame = cap.read()
    if not success:
        print("Error: Could not read from video capture device.")
        return
    
    height, width = frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Initialize components
    ball_tracker = BallTracker()
    trajectory_predictor = TrajectoryPredictor()
    stick_figure = StickFigure(width, height)
    
    # Game state variables
    score = 0
    misses = 0
    ball_caught = False
    prediction_point = None
    
    while True:
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        
        # Flip the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)
        
        # Detect the ball
        ball_pos = ball_tracker.detect_ball(frame)
        
        # Draw the ball on the frame
        if ball_pos:
            frame = ball_tracker.draw_ball(frame, ball_pos)
            
            # Get ball positions for trajectory prediction
            positions = ball_tracker.get_positions()
            
            # Get current ball x-position for stick figure
            current_x = ball_pos[0]
            
            # If we have enough positions, try to predict the trajectory
            if len(positions) >= 5:
                if trajectory_predictor.fit_trajectory(positions):
                    # Get the paddle y-level
                    _, _, paddle_y = stick_figure.get_paddle_bounds()
                    
                    # Predict where the ball will intersect with the paddle y-level
                    intersection = trajectory_predictor.predict_intersection(paddle_y)
                    
                    if intersection:
                        predicted_x, time_to_intersection = intersection
                        prediction_point = (predicted_x, paddle_y)
                        
                        # Only use prediction if it's a reasonable time away
                        # and not too far from current position
                        if time_to_intersection < 15 and abs(predicted_x - current_x) < width/2:
                            # Move stick figure to predicted position
                            stick_figure.move_to(predicted_x)
                        else:
                            # If prediction is too far in the future, just follow the ball
                            stick_figure.move_to(current_x)
                        
                        # Check if ball is near the paddle level
                        if ball_pos[1] >= paddle_y - 10 and ball_pos[1] <= paddle_y + 10:
                            # Get paddle bounds
                            paddle_left, paddle_right, _ = stick_figure.get_paddle_bounds()
                            
                            # Check if ball is within paddle horizontal bounds
                            if paddle_left <= ball_pos[0] <= paddle_right:
                                if not ball_caught:
                                    score += 1
                                    ball_caught = True
                            else:
                                if not ball_caught:
                                    misses += 1
                                    ball_caught = True
                    else:
                        # If no valid intersection is predicted, follow the current ball position
                        stick_figure.move_to(current_x)
                else:
                    # If trajectory prediction fails, just follow the ball
                    stick_figure.move_to(current_x)
            else:
                # Not enough positions for prediction, just follow the ball
                stick_figure.move_to(current_x)
        else:
            # Reset ball_caught when no ball is detected
            ball_caught = False
        
        # Draw the stick figure
        frame = stick_figure.draw(frame)
        
        # Draw prediction point
        if prediction_point and isinstance(prediction_point, tuple) and len(prediction_point) == 2 and isinstance(prediction_point[0], int) and isinstance(prediction_point[1], int):
            cv2.circle(frame, prediction_point, 5, (0, 255, 255), -1)
        
        # Draw score
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Misses: {misses}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Ping Pong Ball Tracker", frame)
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
