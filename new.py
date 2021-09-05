import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
pTime = 0

draw = mp.solutions.drawing_utils
facmesh = mp.solutions.face_mesh		# detect face  #(max_num_faces=1)
face = facmesh.FaceMesh(static_image_mode=False, min_tracking_confidence=0.6, min_detection_confidence=0.6)
# face = facmesh.FaceMesh(Chế độ ảnh tĩnh = True, độ tin cậy theo dõi tối thiểu = 0.6, độ tin cậy phát hiện tối thiểu = 0.6, số mặt tối đa = 1 )                        

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face.process(imgRGB)
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            draw.draw_landmarks(img, faceLms, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)) #, facmesh.FACE_CONNECTIONS

            for id,lm in enumerate(faceLms.landmark):
                print(lm)      # in ra (x, y, z)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                # print(id, x, y)                     # (điểm số , tọa độ x, tọa độ y)
                # if id == 0 or id == 1 or id == 2 or id == 3 or id == 4 or id == 5 or id == 6:
                cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5 , (0, 255, 0), 1) # in ra từ điểm 1 đến 468 trên khuôn mặt


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break





	# _, frm = cap.read()
	# # print(frm.shape)
	# rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	# op = face.process(rgb)
	# # print(op.multi_face_landmarks)					# in ra 468 điểm và mỗi điểm chứa x,y,z

	# if op.multi_face_landmarks:
	# 	for i in op.multi_face_landmarks:
	# 		print(i.landmark[0].y*480)
	# 		# draw.draw_landmarks(frm, i, facmesh.FACE_CONNECTIONS)
	# 		draw.draw_landmarks(frm, i, facmesh.FACE_CONNECTIONS, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=0))


	# cv2.imshow("window", frm)

	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	cap.release()
	# 	cv2.destroyAllWindows()
	# 	break