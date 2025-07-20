import cv2
import numpy as np

class StickFigure:
    def __init__(self, screen_width, screen_height, size=80):
        """
        Initialize the stick figure with screen dimensions and figure size.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = size
        self.paddle_width = size * 0.8  # Paddle width as a fraction of figure size
        
        # Initial position (bottom center of the screen)
        self.x = screen_width // 2
        self.y = screen_height - size // 2
        
        # Paddle y-position (where the ball needs to hit)
        self.paddle_y = self.y - size // 2
        
        # Movement smoothing
        self.target_x = self.x
        self.smoothing_factor = 0.2  # Lower value = smoother movement
    
    def move_to(self, x):
        """
        Move the stick figure to the specified x-coordinate with smoothing.
        """
        # Set target position
        padding = int(self.size / 2)
        self.target_x = max(padding, min(x, self.screen_width - padding))
        
        # Apply smoothing - gradually move toward target
        self.x = int(self.x + self.smoothing_factor * (self.target_x - self.x))
    
    def draw(self, frame):
        """
        Draw the stick figure on the frame.
        """
        # Head
        head_radius = int(self.size / 6)
        head_center = (self.x, self.y - int(self.size / 3))
        cv2.circle(frame, head_center, head_radius, (255, 255, 255), 2)
        
        # Body
        body_start = head_center
        body_end = (self.x, self.y)
        cv2.line(frame, body_start, body_end, (255, 255, 255), 2)
        
        # Arms
        arm_length = int(self.size / 3)
        left_arm_end = (self.x - arm_length, self.y - int(self.size / 6))
        right_arm_end = (self.x + arm_length, self.y - int(self.size / 6))
        cv2.line(frame, (self.x, body_start[1] + head_radius), left_arm_end, (255, 255, 255), 2)
        cv2.line(frame, (self.x, body_start[1] + head_radius), right_arm_end, (255, 255, 255), 2)
        
        # Legs
        leg_length = int(self.size / 3)
        left_leg_end = (self.x - int(self.size / 4), self.y + leg_length)
        right_leg_end = (self.x + int(self.size / 4), self.y + leg_length)
        cv2.line(frame, body_end, left_leg_end, (255, 255, 255), 2)
        cv2.line(frame, body_end, right_leg_end, (255, 255, 255), 2)
        
        # Paddle (held above head)
        paddle_left = int(self.x - self.paddle_width / 2)
        paddle_right = int(self.x + self.paddle_width / 2)
        cv2.line(frame, (paddle_left, self.paddle_y), (paddle_right, self.paddle_y), (0, 255, 0), 4)
        
        return frame
    
    def get_paddle_bounds(self):
        """
        Return the bounds of the paddle: (left_x, right_x, y)
        """
        paddle_left = int(self.x - self.paddle_width / 2)
        paddle_right = int(self.x + self.paddle_width / 2)
        
        return paddle_left, paddle_right, self.paddle_y
