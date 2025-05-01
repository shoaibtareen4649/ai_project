import cv2
import numpy as np
import pickle
import os
import time
from hand_detection import HandDetector

class GestureRecognizer:
    def __init__(self, model_path='gesture_model.pkl', encoder_path='label_encoder.pkl'):
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Run train_model.py first.")
            
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file '{encoder_path}' not found. Run train_model.py first.")
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the label encoder
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize hand detector
        self.detector = HandDetector(detection_confidence=0.7)
        
        # Track quality metrics
        self.hand_position_quality = 0  # 0-100
        self.consecutive_poor_detections = 0
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.0  # seconds
        self.last_position = None
        
    def evaluate_hand_position(self, img, landmark_list):
        """Evaluate the quality of hand position for recognition"""
        quality = 0
        feedback = []
        h, w, c = img.shape
        
        if not landmark_list:
            return 0, ["No hand detected", "Move your hand into the frame"]
        
        # Extract x,y coordinates
        coords = [[lm[1], lm[2]] for lm in landmark_list]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        # Check hand centering
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Calculate distance from image center (0-50 quality score)
        dist_from_center = np.sqrt(((center_x - w/2)/(w/2))**2 + ((center_y - h/2)/(h/2))**2)
        centering_score = max(0, 50 * (1 - dist_from_center))
        
        # Check if hand is too close to edge (at least 10% margin)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate bounding box size (0-30 quality score)
        box_width = x_max - x_min
        box_height = y_max - y_min
        box_ratio = (box_width * box_height) / (w * h)
        size_score = 0
        
        if box_ratio < 0.05:  # Too small
            size_score = 15 * (box_ratio / 0.05)
            feedback.append("Hand is too small in frame")
            feedback.append("Move your hand closer to the camera")
        elif box_ratio > 0.5:  # Too large
            size_score = 30 * (1 - (box_ratio - 0.5) / 0.5)
            feedback.append("Hand is too close to camera") 
            feedback.append("Move your hand further from camera")
        else:  # Good size
            size_score = 30 * (1 - abs(box_ratio - 0.25) / 0.25)
        
        # Check if hand is near edges (0-20 quality score)
        edge_margin_x = min(x_min / w, (w - x_max) / w)
        edge_margin_y = min(y_min / h, (h - y_max) / h)
        
        if edge_margin_x < 0.1 or edge_margin_y < 0.1:
            edge_score = 20 * (min(edge_margin_x, edge_margin_y) / 0.1)
            
            # Directional guidance
            if x_min / w < 0.1:
                feedback.append("Hand too far left")
                feedback.append("Move hand right")
            elif (w - x_max) / w < 0.1:
                feedback.append("Hand too far right")
                feedback.append("Move hand left")
                
            if y_min / h < 0.1:
                feedback.append("Hand too high")
                feedback.append("Move hand down")
            elif (h - y_max) / h < 0.1:
                feedback.append("Hand too low")
                feedback.append("Move hand up")
        else:
            edge_score = 20
        
        # Calculate overall quality score
        quality = int(centering_score + size_score + edge_score)
        
        # If no specific feedback but quality is still low, give general advice
        if quality < 70 and not feedback:
            feedback.append("Adjust hand position")
            feedback.append("Center hand in frame")
        
        return quality, feedback

    def get_placement_guidance(self, img, landmark_list):
        """Generate specific guidance for optimal hand placement"""
        h, w, c = img.shape
        guidance_img = img.copy()
        
        # If no hand detected, show a flashing "Place hand here" indicator
        if not landmark_list:
            # Flashing rectangle (changes opacity based on time)
            opacity = 0.3 + 0.5 * abs(np.sin(time.time() * 3))
            overlay = guidance_img.copy()
            cv2.rectangle(overlay, (int(w*0.2), int(h*0.1)), (int(w*0.8), int(h*0.9)), (0, 255, 255), -1)
            cv2.addWeighted(overlay, opacity, guidance_img, 1-opacity, 0, guidance_img)
            
            # Add text
            cv2.putText(guidance_img, "PLACE HAND HERE", (int(w*0.25), int(h*0.5)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(guidance_img, "PLACE HAND HERE", (int(w*0.25), int(h*0.5)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return guidance_img
            
        # Extract coordinates for analysis
        coords = [[lm[1], lm[2]] for lm in landmark_list]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate center of hand
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Target center is center of frame
        target_x, target_y = w/2, h/2
        
        # Calculate box size (diagonal length)
        box_width = x_max - x_min
        box_height = y_max - y_min
        box_size = np.sqrt(box_width**2 + box_height**2)
        
        # Ideal size is about 40% of frame diagonal
        ideal_size = 0.4 * np.sqrt(w**2 + h**2)
        
        # Create directional guides
        # Draw target zone (ideal position)
        ideal_width = int(0.4 * w)
        ideal_height = int(0.4 * h)
        cv2.rectangle(guidance_img, 
                    (int(target_x - ideal_width/2), int(target_y - ideal_height/2)),
                    (int(target_x + ideal_width/2), int(target_y + ideal_height/2)),
                    (0, 255, 0), 2)
        
        # Draw current hand bounding box
        cv2.rectangle(guidance_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        
        # Distance between current center and ideal center
        dist_x = center_x - target_x
        dist_y = center_y - target_y
        
        # Direction guidance
        arrows = []
        
        # Horizontal guidance
        if dist_x > 50:  # Too far right
            arrows.append((target_x + 100, target_y, target_x + 50, target_y, (0, 0, 255), "MOVE LEFT"))
        elif dist_x < -50:  # Too far left
            arrows.append((target_x - 100, target_y, target_x - 50, target_y, (0, 0, 255), "MOVE RIGHT"))
            
        # Vertical guidance
        if dist_y > 50:  # Too low
            arrows.append((target_x, target_y + 100, target_x, target_y + 50, (0, 0, 255), "MOVE UP"))
        elif dist_y < -50:  # Too high
            arrows.append((target_x, target_y - 100, target_x, target_y - 50, (0, 0, 255), "MOVE DOWN"))
            
        # Size guidance
        size_ratio = box_size / ideal_size
        if size_ratio < 0.7:  # Too small
            cv2.putText(guidance_img, "MOVE CLOSER", (int(w*0.35), 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif size_ratio > 1.3:  # Too large
            cv2.putText(guidance_img, "MOVE FARTHER", (int(w*0.35), 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw directional arrows
        for start_x, start_y, end_x, end_y, color, text in arrows:
            # Draw arrow line
            cv2.arrowedLine(guidance_img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 
                         color, 4, cv2.LINE_AA, tipLength=0.5)
            
            # Position text near arrow
            text_x = (start_x + end_x) // 2
            text_y = (start_y + end_y) // 2
            
            # Text position adjustment based on direction
            if "LEFT" in text:
                text_x += 40
                text_y -= 20
            elif "RIGHT" in text:
                text_x -= 120
                text_y -= 20
            elif "UP" in text:
                text_y += 40
            elif "DOWN" in text:
                text_y -= 40
            
            # Draw text with contrasting outline for visibility
            cv2.putText(guidance_img, text, (int(text_x), int(text_y)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
            cv2.putText(guidance_img, text, (int(text_x), int(text_y)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return guidance_img
    
    def run(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam. Please check your camera connection.")
            return
            
        print("Gesture recognition started.")
        print("Press 'q' to quit")
        
        # For FPS calculation
        prev_frame_time = 0
        new_frame_time = 0
        
        # For gesture smoothing
        gesture_history = []
        confidence_history = []
        max_history = 5
        
        # Get frame dimensions for positioning guides
        success, img = cap.read()
        if success:
            h, w, c = img.shape
        else:
            h, w = 480, 640
            
        # Add toggle flag for guidance mode
        show_guidance = True
        last_toggle_time = 0
        toggle_cooldown = 0.5  # seconds
        
        try:
            while True:
                success, img = cap.read()
                if not success:
                    print("Failed to capture frame from camera")
                    continue
                
                # Calculate FPS
                new_frame_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                
                # Find hands and landmarks
                img = self.detector.find_hands(img)
                landmark_list = self.detector.find_positions(img)
                features = self.detector.get_features(img)
                
                # Apply hand placement guidance if guidance mode is on
                if show_guidance:
                    img = self.get_placement_guidance(img, landmark_list)
                
                # Evaluate hand position quality
                self.hand_position_quality, position_feedback = self.evaluate_hand_position(img, landmark_list)
                
                # Add quality indicator
                quality_color = (0, 0, 255) if self.hand_position_quality < 50 else \
                               (0, 255, 255) if self.hand_position_quality < 70 else \
                               (0, 255, 0)
                               
                cv2.putText(img, f"Position Quality: {self.hand_position_quality}%", 
                          (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
                
                if features:
                    # Reset consecutive poor detection counter
                    self.consecutive_poor_detections = 0
                    
                    # Ensure features have consistent shape
                    if len(features) > 0:
                        # Predict gesture
                        try:
                            prediction_idx = self.model.predict([features])[0]
                            prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
                            confidence = max(self.model.predict_proba([features])[0])
                            
                            # Add to history for smoothing
                            gesture_history.append(prediction)
                            confidence_history.append(confidence)
                            if len(gesture_history) > max_history:
                                gesture_history.pop(0)
                                confidence_history.pop(0)
                                
                            # Get most common gesture in history
                            from collections import Counter
                            smoothed_prediction = Counter(gesture_history).most_common(1)[0][0]
                            avg_confidence = sum(confidence_history) / len(confidence_history)
                            
                            # Color based on confidence
                            if avg_confidence < 0.5:
                                confidence_color = (0, 0, 255)  # Red for low confidence
                            elif avg_confidence < 0.75:
                                confidence_color = (0, 255, 255)  # Yellow for medium confidence
                            else:
                                confidence_color = (0, 255, 0)  # Green for high confidence
                            
                            # Display result
                            cv2.putText(img, f"Gesture: {smoothed_prediction}", 
                                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, confidence_color, 2)
                            cv2.putText(img, f"Confidence: {avg_confidence:.2f}", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, confidence_color, 2)
                            
                            # Give feedback on low confidence
                            if avg_confidence < 0.6 and time.time() - self.last_feedback_time > self.feedback_cooldown:
                                cv2.putText(img, "Low confidence - try adjusting your hand", 
                                          (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                          
                                # Based on the gesture, give specific advice
                                if self.hand_position_quality >= 70:  # Good position but low confidence
                                    cv2.putText(img, "Try a more deliberate gesture", 
                                          (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                          
                                self.last_feedback_time = time.time()
                                
                        except Exception as e:
                            cv2.putText(img, f"Error: {str(e)}", 
                                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # No hand detected
                    self.consecutive_poor_detections += 1
                    
                    # Reset histories when hand disappears
                    if len(gesture_history) > 0:
                        gesture_history = []
                        confidence_history = []
                    
                    # Display guidance after brief delay to avoid flickering
                    if self.consecutive_poor_detections > 5:
                        cv2.putText(img, "No hand detected", 
                                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(img, "Place your hand in the frame", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display position feedback if quality is poor
                if self.hand_position_quality < 70:
                    y_pos = 200
                    for fb in position_feedback[:2]:  # Show max 2 feedback messages
                        cv2.putText(img, fb, 
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        y_pos += 30
                        
                # Display FPS
                cv2.putText(img, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Display recognition status indicator
                status_color = (0, 0, 255)  # Default red
                if len(confidence_history) > 0:
                    avg_conf = sum(confidence_history) / len(confidence_history)
                    if avg_conf > 0.75 and self.hand_position_quality >= 70:
                        status_color = (0, 255, 0)  # Green for good recognition
                    elif avg_conf > 0.5 or self.hand_position_quality >= 50:
                        status_color = (0, 255, 255)  # Yellow for medium recognition
                
                # Draw status indicator circle
                cv2.circle(img, (w - 50, 70), 15, status_color, -1)
                
                # Toggle guidance with 'g' key
                cv2.putText(img, "Press 'g' to toggle guidance", (10, h - 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                cv2.imshow("Gesture Recognition", img)
                
                # IMPROVED KEY DETECTION
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or q/Q
                    print("Exit key pressed - terminating...")
                    break
                
                # Toggle guidance mode with 'g' key
                current_time = time.time()
                if key == ord('g') and current_time - last_toggle_time > toggle_cooldown:
                    show_guidance = not show_guidance
                    last_toggle_time = current_time
                    print(f"Guidance mode: {'ON' if show_guidance else 'OFF'}")
        
        finally:
            # ENHANCED CLEANUP - Ensure everything closes properly
            print("Closing camera and windows...")
            cap.release()
            cv2.destroyAllWindows()
            
            # Force OpenCV to clean up windows
            for i in range(5):
                cv2.waitKey(1)
            
            print("Gesture recognition stopped.")
            return

def main():
    try:
        recognizer = GestureRecognizer()
        recognizer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_model.py to train a model before running gesture recognition.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Force cleanup of any remaining OpenCV windows
        print("Cleaning up...")
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)
        print("Done.")

if __name__ == "__main__":
    main()