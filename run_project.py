import os
import sys
import subprocess
import time
import cv2
import numpy as np

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        import numpy
        import sklearn
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")

def display_hand_guidance():
    """Display guidance for optimal hand placement"""
    print("\n" + "=" * 50)
    print("HAND PLACEMENT GUIDELINES")
    print("=" * 50)
    print("• Position your hand 12-18 inches (30-45 cm) from the camera")
    print("• Ensure your entire hand is visible in the frame")
    print("• Use good, even lighting - avoid harsh shadows")
    print("• Place your hand against a plain background")
    print("• Keep your hand centered in the camera view")
    print("• Make deliberate, clear gestures")
    print("• Rotate your hand slightly for better landmark detection")
    print("=" * 50)
    print("The camera window will open automatically.")
    print("Press any key to continue...")
    input()

def preview_camera(seconds=5):
    """Open camera preview to help user position their hand"""
    print("\nOpening camera preview to help you position your hand...")
    print("Press 'q' to exit preview")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        cap.release()
        return False
    
    height, width = frame.shape[:2]
    
    # Create guidance overlay
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw hand position guide
        cv2.rectangle(frame, 
                      (int(width*0.25), int(height*0.15)), 
                      (int(width*0.75), int(height*0.85)), 
                      (0, 255, 0), 2)
        
        # Add instruction text
        cv2.putText(frame, "Position hand inside rectangle", 
                   (int(width*0.1), 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "Keep hand open with fingers spread", 
                   (int(width*0.1), height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add countdown if using timed preview
        elapsed = time.time() - start_time
        if seconds > 0:
            remaining = max(0, int(seconds - elapsed))
            cv2.putText(frame, f"Preview: {remaining}s", 
                       (width-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if elapsed > seconds:
                break
        
        # Display the frame
        cv2.imshow("Hand Position Guide", frame)
        
        # IMPROVED KEY DETECTION
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or q/Q
            print("Preview stopped by user")
            break
    
    # After the loop, ensure proper cleanup
    print("Closing camera preview...")
    cap.release()
    cv2.destroyAllWindows()

    # Force OpenCV to clean up windows
    for i in range(5):
        cv2.waitKey(1)
    return True

def main():
    """Main project runner"""
    # Check dependencies
    check_dependencies()
    
    # Print welcome message
    print("=" * 50)
    print("Hand Gesture Recognition System")
    print("=" * 50)
    print("\nThis program allows you to:")
    print("1. Collect new gesture data (replaces existing data)")
    print("2. Add more samples to existing gesture data")
    print("3. Train a gesture recognition model")
    print("4. Run real-time gesture recognition")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1' or choice == '2':
            # Display hand placement guidance
            display_hand_guidance()
            
            # Open camera preview to help position hand
            preview_camera(5)
            
            if choice == '1':
                print("\nStarting fresh data collection...")
                # Run data collection script with append_mode=False
                try:
                    from data_collection import main as collect_data
                    collect_data(append_mode=False)
                except Exception as e:
                    print(f"Error during data collection: {e}")
            else:
                print("\nAdding to existing gesture data...")
                # Run data collection script with append_mode=True
                try:
                    from data_collection import main as collect_data
                    collect_data(append_mode=True)
                except Exception as e:
                    print(f"Error during data collection: {e}")
            
        elif choice == '3':
            print("\nStarting model training...")
            # Run training script (uses all available data)
            try:
                from train_model import train_gesture_classifier
                train_gesture_classifier()
            except Exception as e:
                print(f"Error during model training: {e}")
            
        elif choice == '4':
            # Check if model exists before running recognition
            if not os.path.exists("gesture_model.pkl") or not os.path.exists("label_encoder.pkl"):
                print("\nError: Model files not found.")
                print("Please train the model first (option 3).")
                continue
            
            # Display hand placement guidance
            display_hand_guidance()
            
            # Open camera preview to help position hand
            preview_camera(5)
                
            print("\nStarting gesture recognition...")
            print("Follow the on-screen guidance for optimal recognition")
            print("Keep your hand centered and clearly visible")
            print("Try different angles if recognition confidence is low")
            
            # Run recognition script
            try:
                from gesture_recognition import main as run_recognition
                run_recognition()
            except Exception as e:
                print(f"Error during gesture recognition: {e}")
            
        elif choice == '5':
            print("\nExiting program. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1-5.")

if __name__ == "__main__":
    main()