import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False  # Disable the fail-safe
import math
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Initialize MediaPipe Hands for hand gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Global variables
finger_count = 0
prev_x, prev_y = 0, 0
mode = 'none'  # Mode can be 'none', 'finger_count', 'air_mouse', or 'gesture_detection'

# Gesture detection states
gestures = {"peace": False, "thumbs_up": False, "fist": False, "open_hand": False, "pointing": False}

def count_fingers(hand_landmarks):
    """ Count the number of fingers based on the hand landmarks. """
    fingers = 0
    if hand_landmarks[4].y < hand_landmarks[3].y:
        fingers += 1
    if hand_landmarks[8].y < hand_landmarks[7].y:
        fingers += 1
    if hand_landmarks[12].y < hand_landmarks[11].y:
        fingers += 1
    if hand_landmarks[16].y < hand_landmarks[15].y:
        fingers += 1
    if hand_landmarks[20].y < hand_landmarks[19].y:
        fingers += 1
    return fingers

def detect_air_mouse(landmarks):
    """ Use hand landmarks to control mouse position and simulate a tap when thumb and index are close. """
    global prev_x, prev_y

    # Coordinates for the thumb tip (landmark 4) and index finger tip (landmark 8)
    thumb_x = int(landmarks[4].x * 1920)  # Adjust for screen width
    thumb_y = int(landmarks[4].y * 1080)  # Adjust for screen height
    
    index_x = int(landmarks[8].x * 1920)  # Adjust for screen width
    index_y = int(landmarks[8].y * 1080)  # Adjust for screen height

    # Use pyautogui to move the mouse to the new position
    pyautogui.moveTo(index_x, index_y, duration=0.1)

    # Calculate the distance between thumb and index finger (Euclidean distance)
    distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

    # If the distance is small enough (within a threshold), simulate a mouse click
    if distance < 50:  # Threshold distance (adjust if needed)
        pyautogui.click()

    # Calculate the difference in position to determine movement
    dx = index_x - prev_x
    dy = index_y - prev_y

    # Move the mouse based on hand movement
    if abs(dx) > 5 or abs(dy) > 5:
        pyautogui.move(dx, dy)

    prev_x, prev_y = index_x, index_y

def detect_gesture(landmarks):
    """ Detect specific gestures like peace, thumbs up, fist, etc. """
    global gestures
    
    # Peace Gesture: Index and middle fingers up, others closed
    index_finger_up = landmarks[8].y < landmarks[7].y
    middle_finger_up = landmarks[12].y < landmarks[11].y
    other_fingers_closed = landmarks[4].y > landmarks[3].y and landmarks[16].y > landmarks[15].y and landmarks[20].y > landmarks[19].y
    if index_finger_up and middle_finger_up and other_fingers_closed:
        gestures["peace"] = True
    else:
        gestures["peace"] = False
    
    # Thumbs Up Gesture: Thumb up, other fingers closed
    thumb_up = landmarks[4].y < landmarks[3].y
    other_fingers_closed_for_thumb = landmarks[8].y > landmarks[7].y and landmarks[12].y > landmarks[11].y and landmarks[16].y > landmarks[15].y and landmarks[20].y > landmarks[19].y
    if thumb_up and other_fingers_closed_for_thumb:
        gestures["thumbs_up"] = True
    else:
        gestures["thumbs_up"] = False

    # Fist Gesture: All fingers closed
    if all([landmarks[i].y > landmarks[i-1].y for i in [8, 12, 16, 20]]):
        gestures["fist"] = True
    else:
        gestures["fist"] = False

    # Open Hand Gesture: All fingers extended
    if all([landmarks[i].y < landmarks[i-1].y for i in [8, 12, 16, 20]]):
        gestures["open_hand"] = True
    else:
        gestures["open_hand"] = False

    # Pointing Gesture: Only the index finger is extended
    if landmarks[8].y < landmarks[7].y and all([landmarks[i].y > landmarks[i-1].y for i in [4, 12, 16, 20]]):
        gestures["pointing"] = True
    else:
        gestures["pointing"] = False
    
def gen_frames():
    """ Capture frames from the webcam, process them for hand gesture detection and air mouse control. """
    global finger_count, mode, gestures
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a better mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the hand landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the frame
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                if mode == 'finger_count':
                    # Count the fingers
                    finger_count = count_fingers(landmarks.landmark)
                    # Display the finger count on the frame
                    cv2.putText(frame, f'Fingers: {finger_count}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif mode == 'air_mouse':
                    # Detect air mouse movement and tap feature
                    detect_air_mouse(landmarks.landmark)
                elif mode == 'gesture_detection':
                    # Detect gestures like Peace, Thumbs Up, Fist, etc.
                    detect_gesture(landmarks.landmark)
                    if gestures["peace"]:
                        cv2.putText(frame, 'Peace Gesture Detected', (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if gestures["thumbs_up"]:
                        cv2.putText(frame, 'Thumbs Up Gesture Detected', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if gestures["fist"]:
                        cv2.putText(frame, 'Fist Gesture Detected', (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if gestures["open_hand"]:
                        cv2.putText(frame, 'Open Hand Gesture Detected', (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if gestures["pointing"]:
                        cv2.putText(frame, 'Pointing Gesture Detected', (10, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame in JPEG format for rendering
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte string for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<new_mode>')
def set_mode(new_mode):
    """ Set the current mode based on user selection. """
    global mode
    mode = new_mode  # Set the mode to the new mode provided in the URL
    return jsonify(success=True)

@app.route('/finger_count')
def get_finger_count():
    """ Return the current finger count in JSON format. """
    return jsonify(fingers=finger_count)

if __name__ == '__main__':
    app.run(debug=True)
