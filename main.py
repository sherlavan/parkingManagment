# main for manage parking v0.1
import cv2
import streamShow as ss

url = 'rtsp://admin:F31wv32P@192.168.1.68:554/cam/realmonitor?channel=2&subtype=0'
stream = cv2.VideoCapture(url)

ss.showStream(stream)

