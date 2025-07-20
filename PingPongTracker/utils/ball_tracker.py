import cv2
import numpy as np

class BallTracker:
    def __init__(self, lower_color=(20, 100, 100), upper_color=(35, 255, 255)):
        """
        Initialize the ball tracker with color range for a yellow ping pong ball.
        Adjust the color ranges as needed for your specific ball color.
        """
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.positions = []  # Store recent ball positions for trajectory calculation
        self.max_positions = 10  # Maximum number of positions to store
        self.last_pos = None  # Last detected position
        self.kalman = cv2.KalmanFilter(4, 2)  # Kalman filter for tracking
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman_initialized = False

    def detect_ball(self, frame):
        """
        Detect the ping pong ball in the given frame.
        Returns: (x, y, radius) if ball is found, None otherwise
        """
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create a mask for the specified color range
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
            
            # Apply morphological operations to remove noise
            mask = cv2.erode(mask, None, iterations=1)  # Less erosion to preserve small balls
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Apply Gaussian blur to reduce noise further
            blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(blurred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Initialize ball position
            ball_pos = None
            
            # Only proceed if at least one contour was found
            if len(contours) > 0:
                # Sort contours by area, largest first
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Try to find a circular contour
                for c in sorted_contours[:min(5, len(sorted_contours))]:  # Check the 5 largest contours
                    # Calculate circularity
                    area = cv2.contourArea(c)
                    perimeter = cv2.arcLength(c, True)
                    circularity = 0
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Get the minimum enclosing circle
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    
                    # Lower circularity threshold for small balls as they appear less circular when far away
                    min_circularity = 0.5 if radius < 10 else 0.65
                    
                    # Only consider it a ball if the radius is in appropriate range and it's circular enough
                    # Allow smaller radius for distant balls
                    if 3 < radius < 50 and circularity > min_circularity:
                        ball_pos = (int(x), int(y), int(radius))
                        break
            
            # If no ball is found through contour detection but we have a previous position,
            # use Kalman prediction to estimate where it might be
            if ball_pos is None and self.kalman_initialized:
                # Predict ball position using Kalman filter
                prediction = self.kalman.predict()
                pred_x, pred_y = prediction[0, 0], prediction[1, 0]
                
                # Search in a small region around the predicted position
                search_radius = 30
                x_min = max(0, int(pred_x - search_radius))
                y_min = max(0, int(pred_y - search_radius))
                x_max = min(frame.shape[1], int(pred_x + search_radius))
                y_max = min(frame.shape[0], int(pred_y + search_radius))
                
                if x_min < x_max and y_min < y_max:
                    # Extract the region of interest
                    roi = mask[y_min:y_max, x_min:x_max]
                    
                    # Find contours in the ROI
                    roi_contours, _ = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if roi_contours:
                        # Find the largest contour in the ROI
                        largest_contour = max(roi_contours, key=cv2.contourArea)
                        
                        # Get the minimum enclosing circle, adjusting coordinates to original frame
                        ((roi_x, roi_y), radius) = cv2.minEnclosingCircle(largest_contour)
                        x, y = roi_x + x_min, roi_y + y_min
                        
                        if radius > 2:  # Even smaller radius threshold for predicted regions
                            ball_pos = (int(x), int(y), int(radius))
            
            # Update Kalman filter with measurement or prediction
            if ball_pos:
                x, y, _ = ball_pos
                
                # Initialize Kalman filter with first detection
                if not self.kalman_initialized:
                    self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
                    self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
                    self.kalman_initialized = True
                
                # Update Kalman filter with new measurement
                measurement = np.array([[x], [y]], np.float32)
                self.kalman.correct(measurement)
                
                # Add position to history
                self.positions.append((int(x), int(y)))
                
                # Keep only the most recent positions
                if len(self.positions) > self.max_positions:
                    self.positions.pop(0)
                
                self.last_pos = ball_pos
            elif self.kalman_initialized:
                # If no ball detected but Kalman filter initialized, use prediction
                prediction = self.kalman.predict()
                x, y = int(prediction[0, 0]), int(prediction[1, 0])
                
                # Use the last known radius or a default
                radius = self.last_pos[2] if self.last_pos else 10
                
                # Create a "predicted" ball position
                pred_ball_pos = (x, y, radius)
                
                # Add to positions history but mark it as a prediction
                self.positions.append((int(x), int(y)))
                
                # Keep only the most recent positions
                if len(self.positions) > self.max_positions:
                    self.positions.pop(0)
                
                # Return the predicted position
                return pred_ball_pos
            
            return ball_pos
        except Exception as e:
            print(f"Ball detection error: {e}")
            return None
    
    def draw_ball(self, frame, ball_pos):
        """
        Draw the detected ball on the frame.
        """
        if ball_pos:
            x, y, radius = ball_pos
            
            # Draw the ball outline
            cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
            
            # Draw the ball center
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            # Draw ball trail from recent positions
            if len(self.positions) > 1:
                # Draw lines connecting recent positions
                for i in range(1, len(self.positions)):
                    # Calculate color based on position in trail (green to blue)
                    g = int(255 * (1 - i / len(self.positions)))
                    r = 0
                    b = int(255 * (i / len(self.positions)))
                    
                    # Draw line between consecutive positions
                    cv2.line(frame, self.positions[i-1], self.positions[i], (r, g, b), 2)
        
        return frame
    
    def get_positions(self):
        """
        Return the list of recent positions for trajectory calculation.
        """
        return self.positions
