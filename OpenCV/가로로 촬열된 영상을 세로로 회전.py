import cv2
cap = cv2.VideoCapture('city.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('city_output.avi',fourcc, fps * 4, (height, width))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret :
        break

    rotate_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    out.write(rotate_frame)
    cv2.imshow('video',frame)

    if cv2.waitKey(1) == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()