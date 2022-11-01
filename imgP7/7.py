import cv2
cap = cv2.VideoCapture('video/horse.avi')
tmpl = cv2.imread('video/horse_template.jpg', 0)

while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    match = cv2.matchTemplate(gray, tmpl, cv2.TM_SQDIFF_NORMED)
    _, _, min_pos, _ = cv2.minMaxLoc(match)

    tmpl = gray[min_pos[1]:min_pos[1]+tmpl.shape[0], min_pos[0]:min_pos[0]+tmpl.shape[1]]

    cv2.rectangle(frame, min_pos, (min_pos[0] + tmpl.shape[1], min_pos[1] + tmpl.shape[0]), (255, 0, 0), 2)

    cv2.imshow('scene', frame)
    cv2.imshow('template', tmpl)
    ch = cv2.waitKey(30)

    if ch == 27:
        break