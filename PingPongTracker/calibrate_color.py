import cv2
import numpy as np

def main():
    """
    A simple utility to help calibrate HSV color ranges for ping pong ball detection.
    This will display the original image alongside the HSV-filtered image.
    """
    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return
    
    # Create window with trackbars
    cv2.namedWindow("Trackbars")
    
    # Create trackbars for color range adjustment - default values for yellow
    cv2.createTrackbar("L-H", "Trackbars", 20, 179, lambda x: None)
    cv2.createTrackbar("L-S", "Trackbars", 100, 255, lambda x: None)
    cv2.createTrackbar("L-V", "Trackbars", 100, 255, lambda x: None)
    cv2.createTrackbar("U-H", "Trackbars", 35, 179, lambda x: None)
    cv2.createTrackbar("U-S", "Trackbars", 255, 255, lambda x: None)
    cv2.createTrackbar("U-V", "Trackbars", 255, 255, lambda x: None)
    cv2.createTrackbar("Min Radius", "Trackbars", 3, 20, lambda x: None)
    cv2.createTrackbar("Min Circularity", "Trackbars", 50, 100, lambda x: None)
    
    while True:
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        
        # Flip the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get trackbar positions
        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")
        min_radius = cv2.getTrackbarPos("Min Radius", "Trackbars")
        min_circularity = cv2.getTrackbarPos("Min Circularity", "Trackbars") / 100.0
        
        # Create lower and upper HSV bounds
        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])
        
        # Create a mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply morphological operations to remove noise
        mask_processed = cv2.erode(mask, None, iterations=1)
        mask_processed = cv2.dilate(mask_processed, None, iterations=2)
        
        # Apply Gaussian blur for smoother contours
        blurred_mask = cv2.GaussianBlur(mask_processed, (5, 5), 0)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(blurred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(frame, frame, mask=mask)
        detection_result = frame.copy()
        
        # Draw detected balls based on circularity and radius thresholds
        if contours:
            for c in contours:
                # Calculate circularity
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Get the minimum enclosing circle
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                # Draw potential balls with different colors based on detection criteria
                if radius >= min_radius:
                    # Draw the contour
                    cv2.drawContours(detection_result, [c], -1, (0, 255, 0), 2)
                    
                    if circularity >= min_circularity:
                        # Draw a circle for detected balls
                        cv2.circle(detection_result, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                        cv2.putText(detection_result, f"r={int(radius)}, c={circularity:.2f}", 
                                   (int(x) - 40, int(y) - int(radius) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        # Draw a different color for objects that meet radius but not circularity
                        cv2.circle(detection_result, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                        cv2.putText(detection_result, f"r={int(radius)}, c={circularity:.2f}", 
                                   (int(x) - 40, int(y) - int(radius) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Display the original frame, mask, and result
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Processed Mask", blurred_mask)
        cv2.imshow("Color Filter", result)
        cv2.imshow("Detection Result", detection_result)
        
        # Print current HSV range and detection parameters
        print(f"HSV Range: Lower ({l_h}, {l_s}, {l_v}), Upper ({u_h}, {u_s}, {u_v}), Min Radius: {min_radius}, Min Circularity: {min_circularity:.2f}", end="\r")
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final parameters
    print(f"\nFinal Settings:")
    print(f"HSV Range: Lower ({l_h}, {l_s}, {l_v}), Upper ({u_h}, {u_s}, {u_v})")
    print(f"Min Radius: {min_radius}")
    print(f"Min Circularity: {min_circularity:.2f}")
    print("\nUse these values in the BallTracker class:")
    print(f"ball_tracker = BallTracker(lower_color=({l_h}, {l_s}, {l_v}), upper_color=({u_h}, {u_s}, {u_v}))")
    print("\nAnd update the detect_ball method to use:")
    print(f"if {min_radius} < radius < 50 and circularity > {min_circularity:.2f}:")

if __name__ == "__main__":
    main()
