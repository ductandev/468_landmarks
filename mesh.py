import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

facmesh = mp.solutions.face_mesh		# detect face  #(max_num_faces=1)
face = facmesh.FaceMesh(static_image_mode=False,max_num_faces=2, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

while True:

	_, frm = cap.read()
	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = face.process(rgb)
	# print(op.multi_face_landmarks)					# in ra 468 điểm và mỗi điểm chứa x,y,z

	if op.multi_face_landmarks:
		for i in op.multi_face_landmarks:				#print(i) :landmark(x,y,z)
			# print("x = ", i.landmark[0].x*640)
			# print("y = ", i.landmark[0].y*480)
			# print("z = ", i.landmark[0].z*3)
			# draw.draw_landmarks(frm, i, facmesh.FACE_CONNECTIONS)
			draw.draw_landmarks(frm, i, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=0)) #,facmesh.FACE_CONNECTIONS

			for id,lm in enumerate(i.landmark):
				x = i.landmark[id].x*640
				y = i.landmark[id].y*480
				z = i.landmark[id].z*3
				print("x[{}] = {}, y[{}] = {}, z[{}] = {} ".format(id, x, id, y, id, z))

			# break

	# break

	cv2.imshow("window", frm)
	# break

	if cv2.waitKey(1) & 0xFF == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break
