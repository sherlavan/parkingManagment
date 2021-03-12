# main for manage parking v0.1
import cv2
import numpy as np

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


url = 'rtsp://admin:F31wv32P@192.168.1.68:554/cam/realmonitor?channel=2&subtype=0'
url = 'rtsp://admin:F31wv32P@85.202.235.13:23554/Streaming/Channels/101'
stream = cv2.VideoCapture(url)

numberClassifier = cv2.CascadeClassifier('russian_plate_number.xml')
max_w, max_h = stream.get(3), stream.get(4)
print(max_w, max_h)
xc, yc, xc1, yc1 = 600, 200, 1600, 700 # crop coordinate
sigma = 0.33
cv2.namedWindow('stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('stream', 1800, 900)
# cv2.namedWindow('modifed', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('modifed', 800, 600)


while stream.isOpened():
    ret, frame = stream.read()
    cframe = frame[yc:yc1, xc:xc1]
    tframe = four_point_transform(frame, np.array([(650, 300), (1500, 500), (650, 700), (1500, 300)]))
    gray = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)


    plaques = numberClassifier.detectMultiScale(gray, 1.3, 5)
    for i, (x, y, w, h) in enumerate(plaques):
        roi_color = frame[y:y + h, x:x + w]
        cv2.putText(frame, str(x) + " " + str(y) + " " + str(w) + " " + str(h), (480, 220), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255))
        r = 400.0 / roi_color.shape[1]
        dim = (400, int(roi_color.shape[0] * r))
        resized = cv2.resize(roi_color, dim, interpolation=cv2.INTER_AREA)
        w_resized = resized.shape[0]
        h_resized = resized.shape[1]

        frame[100:100 + w_resized, 100:100 + h_resized] = resized

    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # v = np.median(gray)
    #
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    #
    # edge = cv2.Canny(gray, lower, upper)

    cv2.imshow('stream', tframe)
    # cv2.imshow('modifed', resized)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
stream.release()
cv2.destroyAllWindows()





