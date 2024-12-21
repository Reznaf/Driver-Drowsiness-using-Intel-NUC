import cv2
import dlib
import numpy as np
import time
import pygame
import math
from scipy.spatial.distance import euclidean

def estimate_head_pose(landmarks, frame):
        """
        Estimate 3D head pose using facial landmarks
        Returns [pitch, yaw, roll] in degrees
        """
        # 3D model points (simplified)
        model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # 2D image points from landmarks
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),    # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),      # Chin
            (landmarks.part(45).x, landmarks.part(45).y),    # Left eye left corner
            (landmarks.part(36).x, landmarks.part(36).y),    # Right eye right corner
            (landmarks.part(54).x, landmarks.part(54).y),    # Left Mouth corner
            (landmarks.part(48).x, landmarks.part(48).y)     # Right mouth corner
        ], dtype=np.float64)

        # Camera matrix (assuming standard intrinsics)
        size = frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype=np.float64)

        # Distortion coefficients
        dist_coeffs = np.zeros((4,1))

        # Solve PnP to get rotation and translation vectors
        _, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract Euler angles
        sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  
                       rotation_matrix[1,0] * rotation_matrix[1,0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rotation_matrix[2,1] , rotation_matrix[2,2])
            y = math.atan2(-rotation_matrix[2,0], sy)
            z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = math.atan2(-rotation_matrix[2,0], sy)
            z = 0
        
        a = math.degrees(x)
        b = math.degrees(y)
        c = math.degrees(z)

        if abs(c) > 50:
            return [
                abs(a),   # Pitch
                abs(b),   # Yaw
                abs(180.0 - abs(c))
            ] 
        else :
            return [
                    abs(a),   # Pitch
                    abs(b),   # Yaw
                    abs(c)
                ]
    
def preprocess_image(frame):
    """
    Preprocess the image by converting to grayscale and applying adaptive thresholding
    
    Args:
        frame (numpy.ndarray): Input color image
    
    Returns:
        numpy.ndarray: Preprocessed grayscale image with adaptive thresholding
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive Gaussian thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 4))
    preprocessed = clahe.apply(gray)
    
    return preprocessed

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) from eye landmarks
    """
    # Vertical eye landmarks
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal eye landmark
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_gaze_ear(ear_value, head_pose):
    x = head_pose[0]
    y = head_pose[1]
    z = head_pose[2]
    a = 1.04817836
    b = 0.54817836
    c = 0.57624035
    ear = ear_value * (((x*a)+(y*b)+(z*c))/(x+y+z))
    return ear

def ear_variance(params, ear_values, head_pose_values):
    a, b, c = params
    adjusted_ears = []
    for ear, (x, y, z) in zip(ear_values, head_pose_values):
        adjustment_factor = (a * x + b * y + c * z) / (x + y + z)
        adjusted_ear = ear * adjustment_factor
        adjusted_ears.append(adjusted_ear)
    # Calculate variance of adjusted EARs
    return np.var(adjusted_ears)

def calculate_modified_ear_threshold(cap, detector, predictor):
    """
    Calculate EAR threshold by sampling first 10 seconds of video
    Threshold = (max EAR + min EAR) / 2
    """
    # Left and right eye landmark indices
    right_eye_indices = [36, 37, 38, 39, 40, 41]
    left_eye_indices = [42, 43, 44, 45, 46, 47]

    ear_values = []
    start_time = time.time()
    max_ear = float('-inf')
    min_ear = float('inf')

    while time.time() - start_time < 10:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = preprocess_image(frame)

        # Detect faces
        faces = detector(gray)

        # Calibration info
        time_elapsed = time.time() - start_time
        progress = int((time_elapsed / 10) * 100)

        # Clear frame
        frame_display = frame.copy()
        cv2.putText(frame_display, f"Calibrating: {progress}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame_display, f"Current Max EAR: {max_ear:.4f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame_display, f"Current Min EAR: {min_ear:.4f}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Blink Detection Calibration", frame_display)
        cv2.waitKey(1)

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            head_pose = estimate_head_pose(landmarks, frame)

            # Extract eye landmarks
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in left_eye_indices])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                  for i in right_eye_indices])

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            for n in right_eye_indices + left_eye_indices:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0

            avg_ear = calculate_gaze_ear(avg_ear, head_pose)
            
            ear_values.append(avg_ear)

                # Update max and min EAR
            max_ear = max(max_ear, avg_ear)
            min_ear = min(min_ear, avg_ear)

    cv2.destroyWindow("Blink Detection Calibration")

    # Calculate threshold as (max + min) / 2
    if ear_values:
        ear_threshold = (max_ear + min_ear) / 2.0
        return ear_threshold
    else:
        return 0.2

def detect_blinks_and_drowsiness():
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"models\dlib\shape_predictor_68_face_landmarks.dat")
    pygame.mixer.init()
    # Left and right eye landmark indices
    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]

    # Video capture
    set_fps = 10
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, set_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    

    # Calculate adaptive EAR threshold
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start of video
    ear_threshold = calculate_modified_ear_threshold(cap, detector, predictor)
    # ear_threshold = 0.2
    print(f"Calculated EAR Threshold: {ear_threshold:.4f}")

    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Blink tracking variables
    total_blinks = 0
    current_blink_state = False
    blink_frames = 0
    current_classification = "No Blink"
    current_ear = 0.0

    # Running time tracking
    program_start_time = time.time()
    frame_count = 0

    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Calculate running time
        running_time = time.time() - program_start_time
        hours, rem = divmod(int(running_time), 3600)
        minutes, seconds = divmod(rem, 60)

        # Convert to grayscale for face detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = preprocess_image(frame)

        # Detect faces
        faces = detector(gray)

        # Default classification if no face detected
        if len(faces) == 0:
            current_classification = "No Face Detected"
            current_ear = 0.0
            
            # Display information
            cv2.putText(frame, f"Threshold: {ear_threshold:.4f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Classification: {current_classification}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {current_ear:.4f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Total Blinks: {total_blinks}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Running time display
            cv2.putText(frame, f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            cv2.imshow("Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            
            # Extract eye landmarks
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in left_eye_indices])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                  for i in right_eye_indices])

            # Estimate head pose
            head_pose = estimate_head_pose(landmarks, frame)

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            for n in right_eye_indices + left_eye_indices:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
            # Average EAR
            current_ear = (left_ear + right_ear) / 2.0

            current_ear = calculate_gaze_ear(current_ear, head_pose)

            # Blink detection logic
            if current_ear < ear_threshold:
                if not current_blink_state:
                    current_blink_state = True
                    blink_frames = 1
                else:
                    blink_frames += 1
            
                # Blink classification
                if current_blink_state:
                    # Determine blink duration in frames
                    if 1 <= blink_frames <= 5:  # Adjust these values based on your fps
                        total_blinks += 1
                        current_classification = "Normal Blink"
                    elif blink_frames > 5:
                        total_blinks += 1
                        current_classification = "Long Blink"
                        pygame.mixer.music.load(r"assets/sounds/kantuk.mp3")
                        pygame.mixer.music.play()
        
                else:
                    current_classification = "No Blink"
            else:
                current_blink_state = False
                blink_frames = 0
            
            # Display head pose and gaze information
            cv2.putText(frame, f"Head Pose (Pitch, Yaw, Roll): {head_pose}", 
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 2)

            # Continuous classification display
            cv2.putText(frame, f"Threshold: {ear_threshold:.4f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Classification: {current_classification}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # EAR value display
            cv2.putText(frame, f"EAR: {current_ear:.4f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Total blinks display
            cv2.putText(frame, f"Total Blinks: {total_blinks}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Blink Frame: {blink_frames}", 
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Running time display
            # cv2.putText(frame, f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}", 
            #             (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            cv2.putText(frame, f"FPS: {fps}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # Display the frame
        cv2.imshow("Blink Detection", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# Run the detection
if __name__ == "__main__":
    detect_blinks_and_drowsiness()