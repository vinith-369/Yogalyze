
from flask import Flask, render_template, Response
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Load model and labels
model_dict = pickle.load(open('trikonasana_model.p', 'rb'))
model = model_dict['model']
labels = model_dict['labels']
labels_dict = {label: label for label in labels}

# Define the sequence of asanas
asana_sequence = ["0", "1", "2", "3"]  # Example sequence
current_asana_index = 0
frame_count = 0
hold_count = 0
frames_required = 30
hold_frames = 30  # Extra hold time between asanas

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not opened")
        return

    while True:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        if not ret:
            break
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            for landmark in results.pose_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            for landmark in results.pose_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
            
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
            
            prediction = model.predict([np.asarray(data_aux)])
            predicted_asana = str(prediction[0])
            
            # Check if predicted asana matches the required one
            if predicted_asana == asana_sequence[current_asana_index]:
                frame_count += 1
                if frame_count >= frames_required:
                    if hold_count < hold_frames:
                        hold_count += 1
                    else:
                        frame_count = 0
                        hold_count = 0
                        current_asana_index += 1
                        if current_asana_index >= len(asana_sequence):
                            print("Training completed!")
                            break
            else:
                frame_count = 0  # Reset if incorrect pose
                hold_count = 0
            
            # Show only correct asana
            if frame_count > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, predicted_asana, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Show the required asana for guidance
        cv2.putText(frame, f"Perform: {asana_sequence[current_asana_index]}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display frames left for current step
        frames_left = max(frames_required - frame_count, 0)
        cv2.putText(frame, f"Frames left: {frames_left}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', 
                           asana_sequence=asana_sequence, 
                           current_asana_index=current_asana_index, 
                           frames_left=max(frames_required - frame_count, 0))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)











# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load model and labels
# model_dict = pickle.load(open('yt_v2_model.p', 'rb'))
# model = model_dict['model']
# labels = model_dict['labels']
# labels_dict = {label: label for label in labels}

# # Define the sequence of asanas
# asana_sequence = ["Tadasana", "Tiryak_Tadasana_Left", "Tiryak_Tadasana_Right"]  # Example sequence
# current_asana_index = 0
# frame_count = 0
# hold_count = 0
# frames_required = 30
# hold_frames = 30  # Extra hold time between asanas

# # Start video capture
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Camera not opened")

# # Initialize Mediapipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
    
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             mp_drawing_styles.get_default_pose_landmarks_style()
#         )
        
#         for landmark in results.pose_landmarks.landmark:
#             x_.append(landmark.x)
#             y_.append(landmark.y)
        
#         for landmark in results.pose_landmarks.landmark:
#             data_aux.append(landmark.x - min(x_))
#             data_aux.append(landmark.y - min(y_))
        
#         x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
#         x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
        
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_asana = str(prediction[0])
        
#         # Check if predicted asana matches the required one
#         if predicted_asana == asana_sequence[current_asana_index]:
#             frame_count += 1
#             if frame_count >= frames_required:
#                 if hold_count < hold_frames:
#                     hold_count += 1
#                 else:
#                     frame_count = 0
#                     hold_count = 0
#                     current_asana_index += 1
#                     if current_asana_index >= len(asana_sequence):
#                         print("Training completed!")
#                         break
#         else:
#             frame_count = 0  # Reset if incorrect pose
#             hold_count = 0
        
#         # Show only correct asana
#         if frame_count > 0:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
#             cv2.putText(frame, predicted_asana, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    
#     # Show the required asana for guidance
#     cv2.putText(frame, f"Perform: {asana_sequence[current_asana_index]}", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
#     # Display frames left for current step
#     frames_left = max(frames_required - frame_count, 0)
#     cv2.putText(frame, f"Frames left: {frames_left}", (50, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
#     cv2.imshow('frame', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Training completed!")