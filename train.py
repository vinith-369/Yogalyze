import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('suryanamaskar_data.pickle', 'rb'))


data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('suryanamaskar_model.p', 'wb')
pickle.dump({'model': model, 'labels': list(set(labels))}, f)  # Save labels
f.close()
f.close()




# /////////////////////////////////////////////////////////////////




# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load the trained model
# model_dict = pickle.load(open('suryanamaskar_model.p', 'rb'))
# model = model_dict['model']
# labels = model_dict['labels']  # Load labels
# labels_dict = {label: label for label in labels}  # Identity mapping

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Camera not opened")

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

#         # Predict pose
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = str(prediction[0])  # Fix here

#         print(predicted_character)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)

# cap.release()
# cv2.destroyAllWindows()