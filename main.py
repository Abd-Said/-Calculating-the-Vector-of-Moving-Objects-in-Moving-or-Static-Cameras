import cv2
import numpy as np
import math


farneback_params = dict(pyr_scale=0.5, levels=5, winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
cap = cv2.VideoCapture('C:/Users/user/Desktop/yolcu.mp4')

prev_frame = None

while True:
    cam_vector = [0,0] #decrease the cam vector
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, **farneback_params)

        for y in range(0, frame.shape[0], 30):
            for x in range(0, frame.shape[1], 30):
                fx, fy = flow[y, x] - cam_vector
                if abs(fx+fy) > 2:
                    cv2.arrowedLine(frame, (x, y), (int(x + fx), int(y + fy)), (0, 0, 255), thickness=3)
                    print(f"obj detected in: {x}, {y}")
                    speed = math.sqrt(fx**2 + fy**2)
                    print(speed)
                else:
                    cv2.arrowedLine(frame, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), thickness=3)




    prev_frame = gray

    cv2.imshow('Optical Flow', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
