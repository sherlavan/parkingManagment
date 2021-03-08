import cv2
import numpy as np

def showStream(stream):
    sigma = 0.33
    cv2.namedWindow('stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('stream', 800, 600)
    cv2.namedWindow('modifed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('modifed', 800, 600)

    while stream.isOpened():
        ret, frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        v = np.median(gray)

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        edge = cv2.Canny(gray, 30, 100)
        cv2.imshow('stream', frame)
        cv2.imshow('modifed', edge)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    stream.release()
    cv2.destroyAllWindows()