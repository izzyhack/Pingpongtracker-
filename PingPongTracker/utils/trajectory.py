import numpy as np
from scipy.optimize import curve_fit

class TrajectoryPredictor:
    def __init__(self):
        """
        Initialize the trajectory predictor.
        """
        self.trajectory_model = None
        self.min_points = 5  # Minimum points needed for prediction
    
    def fit_trajectory(self, positions):
        """
        Fit a quadratic function to the ball positions to model its trajectory.
        """
        if len(positions) < self.min_points:
            return False
        
        # Extract x and y coordinates
        x_coords = np.array([pos[0] for pos in positions])
        y_coords = np.array([pos[1] for pos in positions])
        
        # Create time points (assuming constant time between frames)
        time_points = np.arange(len(positions))
        
        # Fit quadratic functions for x and y coordinates
        try:
            # x(t) = a_x*t^2 + b_x*t + c_x
            self.x_params, _ = curve_fit(self._quadratic_func, time_points, x_coords, maxfev=10000)
            
            # y(t) = a_y*t^2 + b_y*t + c_y
            self.y_params, _ = curve_fit(self._quadratic_func, time_points, y_coords, maxfev=10000)
            
            self.trajectory_model = (self.x_params, self.y_params)
            return True
        except Exception as e:
            print(f"Trajectory fitting error: {e}")
            return False
    
    def _quadratic_func(self, t, a, b, c):
        """
        Quadratic function for curve fitting: f(t) = a*t^2 + b*t + c
        """
        return a * t**2 + b * t + c
    
    def predict_position(self, time_steps_ahead):
        """
        Predict the ball position n time steps ahead.
        """
        if self.trajectory_model is None:
            return None
        
        x_params, y_params = self.trajectory_model
        
        # Current time is the last index of the positions used for fitting
        current_time = self.min_points - 1
        future_time = current_time + time_steps_ahead
        
        # Predict future x and y coordinates
        future_x = self._quadratic_func(future_time, *x_params)
        future_y = self._quadratic_func(future_time, *y_params)
        
        return int(future_x), int(future_y)
    
    def predict_intersection(self, y_level):
        """
        Predict where the ball will intersect with a horizontal line at y_level.
        Returns (x_intersection, time_to_intersection) or None if no intersection.
        """
        if self.trajectory_model is None:
            return None
        
        try:
            x_params, y_params = self.trajectory_model
            
            # Solve the quadratic equation: a*t^2 + b*t + c = y_level
            a, b, c = y_params
            c_adjusted = c - y_level
            
            # Calculate discriminant
            discriminant = b**2 - 4*a*c_adjusted
            
            if discriminant < 0:
                # No real solutions, ball will not reach y_level
                return None
            
            # Find the two possible times
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
            
            # Choose the positive time that is greater than current time
            current_time = self.min_points - 1
            valid_times = [t for t in [t1, t2] if t > current_time]
            
            if not valid_times:
                return None
            
            # Choose the earliest intersection time
            t_intersection = min(valid_times)
            
            # Calculate the x-coordinate at the intersection time
            x_intersection = self._quadratic_func(t_intersection, *x_params)
            
            # Ensure the intersection point is within screen bounds and is an integer
            x_intersection = int(max(0, min(x_intersection, 640)))  # Assuming max width is 640
            
            # Calculate time to intersection from now
            time_to_intersection = t_intersection - current_time
            
            return x_intersection, time_to_intersection
        except Exception as e:
            print(f"Intersection prediction error: {e}")
            return None
        
        # Calculate time to intersection from now
        time_to_intersection = t_intersection - current_time
        
        return int(x_intersection), time_to_intersection
