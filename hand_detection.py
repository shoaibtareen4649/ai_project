import cv2
import numpy as np

class HandDetector:
    def __init__(self, detection_confidence=0.5, max_hands=2, **kwargs):
        self.detection_confidence = detection_confidence
        self.max_hands = max_hands
        self.contour = None
        
    def find_hands(self, img, draw=True):
        """Detect hands using OpenCV skin detection"""
        # Create a copy of the image
        result_img = img.copy()
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (hand)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Only process if contour is large enough
            if cv2.contourArea(max_contour) > 5000:
                if draw:
                    # Draw contour
                    cv2.drawContours(result_img, [max_contour], 0, (0, 255, 0), 2)
                    
                    # Draw convex hull
                    hull = cv2.convexHull(max_contour)
                    cv2.drawContours(result_img, [hull], 0, (0, 0, 255), 3)
                
                self.contour = max_contour
                self.results = True
                return result_img
        
        self.contour = None
        self.results = False
        return result_img
    
    def find_positions(self, img, hand_no=0):
        """Extract key points from the contour"""
        if not hasattr(self, 'contour') or self.contour is None:
            return []
        
        landmarks = []
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(self.contour)
        
        # Add center point (palm)
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
            
        landmarks.append([0, cx, cy])
        
        # Add corners of bounding box
        landmarks.append([1, x, y])  # Top-left
        landmarks.append([2, x + w, y])  # Top-right
        landmarks.append([3, x, y + h])  # Bottom-left
        landmarks.append([4, x + w, y + h])  # Bottom-right
        
        # Find convexity defects (for fingertips)
        if len(self.contour) >= 5:
            hull_indices = cv2.convexHull(self.contour, returnPoints=False)
            if len(hull_indices) > 3:
                try:
                    defects = cv2.convexityDefects(self.contour, hull_indices)
                    if defects is not None:
                        for i in range(min(16, defects.shape[0])):
                            s, e, f, d = defects[i, 0]
                            if d / 256.0 > 10:
                                # Add points likely to be fingertips or between fingers
                                start_point = tuple(self.contour[s][0])
                                landmarks.append([5 + i, start_point[0], start_point[1]])
                except:
                    pass
                    
        # Ensure we have at least 21 landmarks for consistency with MediaPipe
        contour_len = len(self.contour)
        if contour_len > 0:
            remaining_points = 21 - len(landmarks)
            if remaining_points > 0:
                step = max(1, contour_len // remaining_points)
                idx = 0
                while len(landmarks) < 21 and idx < contour_len:
                    point = self.contour[idx][0]
                    landmarks.append([len(landmarks), point[0], point[1]])
                    idx += step
        
        return landmarks[:21]  # Return at most 21 points
    
    def get_features(self, img, hand_no=0):
        """Extract normalized features for classification"""
        landmark_list = self.find_positions(img, hand_no)
        
        if not landmark_list:
            return []
        
        # Extract x,y coordinates
        coords = [[lm[1], lm[2]] for lm in landmark_list]
        
        # Find bounding box for normalization
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Handle division by zero
        box_width = max(x_max - x_min, 1)
        box_height = max(y_max - y_min, 1)
        
        # Normalize coordinates relative to bounding box (scale invariant)
        normalized_coords = []
        for c in coords:
            nx = (c[0] - x_min) / box_width
            ny = (c[1] - y_min) / box_height
            normalized_coords.extend([nx, ny])
        
        return normalized_coords