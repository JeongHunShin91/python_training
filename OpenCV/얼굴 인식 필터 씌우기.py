# face detection
import cv2
import mediapipe as mp

def overlay(image,x,y,w,h,overlay_imge): # 대상 이미지 3 채널, 덮어씌울 이미지 4채널
    alpha = overlay_imge[:,:,3] # BGRA
    mask_image = alpha / 255 # 1: 불투명 0: 투명

    for c in range(0,3):
        image[y-h:y+h,x-w:x+w,c] = (overlay_imge[:,:,c]*mask_image)+(image[y-h:y+h,x-w:x+w,c]*(1-mask_image))
# 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 모듈
mp_drawing = mp.solutions.drawing_utils # 얼굴의 특징을 그리기 위한 모듈

# 동영상 파일 열기
cap = cv2.VideoCapture('face.mp4')

# 이미지 불러오기
image_right_eye =cv2.imread('right_eye.png',cv2.IMREAD_UNCHANGED)
image_left_eye = cv2.imread('left_eye.png',cv2.IMREAD_UNCHANGED)
image_nose = cv2.imread('nose.png',cv2.IMREAD_UNCHANGED)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.95) as face_detection: # 신뢰도
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            # 6개의 특징 : 오른쪽 눈, 왼쪾눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                # print(detection)

                # 특정 위치 가져오기
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]
                nose_tip = keypoints[2]

                h, w, _ = image.shape# 이미지로부터 세로, 가로 크기

                # 위치 그리기
                right_eye = (int(right_eye.x*w)-20, int(right_eye.y*h)-100)
                left_eye = (int(left_eye.x * w)+20, int(left_eye.y * h)-100)
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

                # cv2.circle(image, right_eye,10,(255,0,0), 10, cv2.LINE_AA)
                # cv2.circle(image, left_eye, 10, (255, 0, 0), 10, cv2.LINE_AA)
                # cv2.circle(image, nose_tip, 20, (0,255,255),10,cv2.LINE_AA)
                #
                # # 각 특징에다 이미지 넣기
                # image[right_eye[1]-50:right_eye[1]+50,right_eye[0]-50:right_eye[0]+50] =image_right_eye
                # image[left_eye[1] - 50:left_eye[1] + 50, left_eye[0] - 50:left_eye[0] + 50] = image_left_eye
                # image[nose_tip[1] - 50:nose_tip[1] + 50, nose_tip[0] - 50:nose_tip[0] + 50] = image_nose
                # 사진붙이기(image,x,y,w,h,overlay_imge)
                overlay(image, *right_eye,50,50,image_right_eye)
                overlay(image, *left_eye, 50, 50, image_left_eye)
                overlay(image, *nose_tip, 100, 100, image_nose)


        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection',image)

        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()