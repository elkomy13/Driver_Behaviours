# %%
# !pip install mediapipe

# %%
import numpy as np
import cv2
import mediapipe as mp

# %%
# Constants
EYE_AR_THRESH_INITIAL = 0.79
EYE_AR_CONSEC_FRAMES = 20
  # Number of consecutive frames the eye must be below the threshold to detect drowsniess

HEAD_POSE_DOWN_THRESH = 90
HEAD_POSE_RIGHT_THRESH = 70

EAR_HISTORY_SIZE = 100


COUNTER = 0
ALARM_ON = False
ear_history = []

# %%
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define indexes for the left and right eye from the MediaPipe landmarks
LEFT_EYE_IDX = [33, 133, 160, 158, 144, 153]  # Left eye landmarks
RIGHT_EYE_IDX = [362, 263, 387, 385, 373, 380]  # Right eye landmarks

# Define indexes for head pose estimation (Nose and Eyes)
NOSE_TIP_IDX = 1
# CHIN_TIP_IDX = 152 

LEFT_EYE_INNER_IDX = 133
RIGHT_EYE_INNER_IDX = 362

# %% [markdown]
# ##  Head Pose Estimation Function

# %%
def get_head_pose(landmarks, frame_shape):

    nose_tip = np.array([landmarks.landmark[NOSE_TIP_IDX].x, landmarks.landmark[NOSE_TIP_IDX].y]) ## extarct the coordinates of the nose from the face 
    
    left_eye_inner = np.array([landmarks.landmark[LEFT_EYE_INNER_IDX].x, landmarks.landmark[LEFT_EYE_INNER_IDX].y]) # extarct the coordinates of the left eye from the face 

    right_eye_inner = np.array([landmarks.landmark[RIGHT_EYE_INNER_IDX].x, landmarks.landmark[RIGHT_EYE_INNER_IDX].y]) # extarct the coordinates of the right from the face 

    # Convert normalized coordinates to pixel coordinates

    # the arrays are in range [0-1] so we multibly it by the frame width and height

    nose_tip *= np.array([frame_shape[1], frame_shape[0]]) 
    left_eye_inner *= np.array([frame_shape[1], frame_shape[0]])
    right_eye_inner *= np.array([frame_shape[1], frame_shape[0]])

    # Compute head pose (simplified version)
    nose_to_eyes_vector = nose_tip - (left_eye_inner + right_eye_inner) / 2
    head_tilt_angle = np.degrees(np.arctan2(nose_to_eyes_vector[1], nose_to_eyes_vector[0]))

    # Calculate bounding box
    x_coords = [landmarks.landmark[idx].x * frame_shape[1] for idx in [33, 263, 1, 62]] # [left , right , top, down]
    y_coords = [landmarks.landmark[idx].y * frame_shape[0] for idx in [33, 263, 1, 62]]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))


    if head_tilt_angle <= HEAD_POSE_DOWN_THRESH and head_tilt_angle > 80:
        head_pose = "Down"
    elif head_tilt_angle < HEAD_POSE_RIGHT_THRESH:
        head_pose = "Right"
    elif head_tilt_angle >= HEAD_POSE_RIGHT_THRESH and head_tilt_angle <= 80:
        head_pose = "Left"
    elif head_tilt_angle > HEAD_POSE_DOWN_THRESH:
        head_pose = "Forward"
    else:
        head_pose = "Unknown"  # Fallback case if none of the conditions match

    
    return head_pose, head_tilt_angle, (x_min, y_min, x_max, y_max)



# %% [markdown]
# # Define the eye aspect ratio function

# %%
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# %%
# cap = cv2.VideoCapture("petal_20240814_134649.mp4")
# cap = cv2.VideoCapture("driver.mp4")
cap = cv2.VideoCapture("one.mp4")


width = 1280
height = 720
slow_down_factor = 2

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (width, height))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks using MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Estimate head pose and bounding box
                head_pose, head_tilt_angle, (x_min, y_min, x_max, y_max) = get_head_pose(face_landmarks, frame.shape)
                
                # Draw bounding box around the head
                cv2.rectangle(frame, (x_min - 50, y_min - 50), (x_max + 50, y_max + 60), (0, 255, 0), 2)
                
                if head_pose == "Forward":

                    # Extract the left and right eye coordinates
                    leftEye = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y) for idx in LEFT_EYE_IDX if idx < len(face_landmarks.landmark)])
                    rightEye = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y) for idx in RIGHT_EYE_IDX if idx < len(face_landmarks.landmark)])
                    
                    # Convert normalized coordinates to pixel coordinates
                    leftEye *= np.array([frame.shape[1], frame.shape[0]])
                    rightEye *= np.array([frame.shape[1], frame.shape[0]])
                    
                    # Draw eye landmarks as circles
                    for idx in range(len(leftEye)):
                        x, y = int(leftEye[idx][0]), int(leftEye[idx][1])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    for idx in range(len(rightEye)):
                        x, y = int(rightEye[idx][0]), int(rightEye[idx][1])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # Compute the eye aspect ratio for both eyes
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    
                    # Average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0

                    # Update EAR history
                    ear_history.append(ear)
                    if len(ear_history) > EAR_HISTORY_SIZE:
                        ear_history.pop(0)
                    
                    # Calculate dynamic threshold
                    if len(ear_history) > 0:
                        ear_mean = np.mean(ear_history)
                        dynamic_threshold = max(EYE_AR_THRESH_INITIAL, ear_mean - 0.1)
                    else:
                        dynamic_threshold = EYE_AR_THRESH_INITIAL
                    
                    # Check if the eye aspect ratio is below the dynamic threshold
                    if ear < dynamic_threshold or ear > 1.1:
                        COUNTER += 1
                        
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            if not ALARM_ON:
                                ALARM_ON = True
                                print("Drowsiness Detected!")
                                
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        COUNTER = 0
                        if ALARM_ON:
                            ALARM_ON = False
                        cv2.putText(frame, "Good job, stay awake", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display the computed eye aspect ratio
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display head tilt angle
                cv2.putText(frame, f"Head Tilt Angle: {head_tilt_angle:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display head pose
                cv2.putText(frame, f"Head Pose: {head_pose}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Check head pose and show alert for left/right
                if head_pose == "Down":
                    cv2.putText(frame, "DROWSINESS ALERT! (Head Down)", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif head_pose == "Left" or head_pose == "Right":
                    cv2.putText(frame, "Focus on the road", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        # Display the frame
        cv2.imshow("Frame", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1000 // (30 // slow_down_factor)) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


# %%


# %%


# %%



