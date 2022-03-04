import cv2
import YOLO
import Detect
capture= cv2.VideoCapture(0)

mask = "No Mask"
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0) 
thickness = 1
if capture.isOpened() is False:
	print("Error opening camera")
i = 0
while True:
	ret, frame = capture.read()
	count=0
	if ret is True:
		detected = YOLO.humanDetect(frame)
		if detected is not None:
			print("Human Detected")
			face = Detect.detect_face(detected)
			filename = "samarth" + str(i) + ".jpg"
			i += 1
			if face is not None:
				if count<=5:
					cv2.imwrite(filename, face)
					count+=1

		else:
			print('Human Not detected')
		frame = cv2.putText(frame, mask, org, font, fontScale, color, thickness, cv2.LINE_AA)

		cv2.imshow("Video", frame)
		if cv2.waitKey(2) & 0xFF == ord('q'): 
			break
	

capture.release()
cv2.destroyAllWindows()
