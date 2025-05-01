import cv2
import numpy as np
import os
import time
from hand_detection import HandDetector

class GestureDataCollector:
    def __init__(self, base_dir="gesture_data"):
        self.base_dir = base_dir
        self.detector = HandDetector(detection_confidence=0.7)
        
        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def collect_gesture_data(self, gesture_name, num_samples=100, append_mode=True):
        # Create directory for gesture if it doesn't exist
        data_dir = f"{self.base_dir}/{gesture_name}"
        os.makedirs(data_dir, exist_ok=True)
        
        # Get starting index for sample naming
        start_idx = 0
        if append_mode:
            existing_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
            if existing_files:
                # Extract numbers from filenames and find the highest
                indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
                start_idx = max(indices) + 1
        else:
            # In non-append mode, clear existing files
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Prepare user with preview before collecting
        print(f"\nPreparing to collect data for: {gesture_name.upper()}")
        print("Please position your hand and prepare to make the gesture.")
        input("Press ENTER when ready to begin preview...")
        
        # Open camera for preview
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return 0
            
        # Preview gesture before collecting
        preview_time = 5  # seconds
        start_time = time.time()
        print(f"Preview mode: Show your {gesture_name} gesture")
        print("Press SPACE to begin collection or 'q' to cancel")
        
        # Get frame dimensions for better text positioning
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            cap.release()
            return 0
            
        height, width = frame.shape[:2]
        
        # Preview loop
        start_collection = False
        while not start_collection:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Create a semi-transparent overlay for instructions
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, height-140), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Show preview guidance
            cv2.putText(frame, f"PREVIEW: {gesture_name.upper()}", 
                      (int(width/2)-180, height-110), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                      
            cv2.putText(frame, "Position your hand and make the gesture", 
                      (int(width/2)-240, height-70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            cv2.putText(frame, "Press SPACE to start collecting data", 
                      (int(width/2)-220, height-40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add a visual guide for hand placement
            cv2.rectangle(frame, 
                        (int(width*0.25), int(height*0.15)), 
                        (int(width*0.75), int(height*0.85)), 
                        (0, 255, 0), 3)
            
            # Process hand for visual feedback
            preview_frame = self.detector.find_hands(frame)
            
            cv2.imshow("Gesture Collection - Preview", preview_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key
                start_collection = True
                print("Beginning data collection...")
            elif key == ord('q'):
                print("Collection canceled by user")
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return 0
        
        # Begin actual collection
        count = 0
        
        # Prepare a countdown before starting
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                # Draw countdown
                cv2.putText(frame, f"Starting in {i}...", 
                           (int(width/2)-150, int(height/2)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                           
                cv2.imshow("Gesture Collection - Preview", frame)
                cv2.waitKey(1)
                time.sleep(1)
        
        print(f"Collection started for {gesture_name}. Collecting {num_samples} samples.")
        print("Maintain the gesture. Press 'q' to stop early.")
        
        # Collection loop
        while count < num_samples:
            success, img = cap.read()
            if not success:
                print("Failed to capture image from camera")
                continue
                
            # Find hands
            img = self.detector.find_hands(img)
            features = self.detector.get_features(img)
            
            if features:
                # Save features to file with new index
                np.save(f"{data_dir}/sample_{start_idx + count}.npy", features)
                count += 1
                
                # Create a more visually appealing overlay for text
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (330, 150), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                
                # Display progress with improved visuals
                cv2.putText(img, f"Collecting: {count}/{num_samples}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
                cv2.putText(img, f"Gesture: {gesture_name}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                           
                # Progress bar
                progress = int((count / num_samples) * 300)
                cv2.rectangle(img, (10, 100), (10 + progress, 120), (0, 255, 0), -1)
                cv2.rectangle(img, (10, 100), (310, 120), (255, 255, 255), 2)
                
                # Add a visual guide for hand placement
                cv2.rectangle(img, 
                            (int(width*0.25), int(height*0.15)), 
                            (int(width*0.75), int(height*0.85)), 
                            (0, 255, 0), 2)
            
            cv2.imshow("Gesture Collection", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Collection stopped early by user")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # This additional waitKey helps ensure windows are properly closed
        
        print(f"\nCollected {count} samples for gesture: {gesture_name}")
        print(f"Total samples for this gesture: {start_idx + count}")
        
        # Wait for user to continue
        input("\nPress ENTER to continue to the next gesture...")
        return count

def main(append_mode=True):
    collector = GestureDataCollector()
    
    # Define gestures to collect
    gestures = ["fist", "open_palm", "peace", "thumbs_up", "point"]
    
    print("\n" + "="*60)
    print("HAND GESTURE DATA COLLECTION".center(60))
    print("="*60)
    
    if append_mode:
        print("\nThis process will add samples to your existing gesture data.")
    else:
        print("\nThis process will replace existing data with new samples.")
        
    print("\nYou will collect data for the following gestures:")
    for i, gesture in enumerate(gestures, 1):
        print(f"  {i}. {gesture.upper()}")
    
    print("\nFor each gesture:")
    print("1. You'll first see a preview to position your hand")
    print("2. Press SPACE to begin collecting samples")
    print("3. Hold the gesture steady while samples are collected")
    print("4. Press ENTER to proceed to the next gesture")
    print("\nYou can press 'q' at any time to stop collection for the current gesture.")
    
    input("\nPress ENTER to begin the process...")
    
    samples_collected = {}
    for gesture in gestures:
        samples = collector.collect_gesture_data(gesture, append_mode=append_mode)
        samples_collected[gesture] = samples
        
        if samples < 10:
            print(f"\nWARNING: Only {samples} samples collected for '{gesture}'.")
            print("This may not be enough for accurate recognition.")
            response = input("Would you like to try collecting more samples for this gesture? (y/n): ")
            if response.lower() == 'y':
                additional = collector.collect_gesture_data(gesture, append_mode=True)
                samples_collected[gesture] += additional
    
    # Summary at the end
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY".center(60))
    print("="*60)
    print(f"\nSamples collected by gesture:")
    for gesture, count in samples_collected.items():
        print(f"  • {gesture.upper()}: {count} samples")
    
    print(f"\nAll data saved in the '{collector.base_dir}' directory")
    print("\nNext steps:")
    print("1. Train your gesture recognition model (Option 3 in main menu)")
    print("2. Run real-time recognition (Option 4 in main menu)")
    
    input("\nPress ENTER to return to the main menu...")

if __name__ == "__main__":
    main()