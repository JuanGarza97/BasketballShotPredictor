import cv2
import numpy as np
import math
from ColorTrackingModule import ColorFinder

VIDEO_NUM = 6
VIDEO_PATH = f'Resources/Videos/video{VIDEO_NUM}.mp4'
IMG_PATH = 'Resources/Ball.png'
# HSV colors of the ball
BALL_HSV = {'h_min': 8, 's_min': 96, 'v_min': 115, 'h_max': 14, 's_max': 255, 'v_max': 255}
# X range of the basket
X_RANGE = [330, 430]
# Y value of the basket
Y_BASKET = 590
# Points needed to do a prediction
POINTS = 10
# Delay between frames
DELAY = 10


def find_ball(img, mask, min_area=1000, threshold=0, draw=True, color=(255, 0, 0)):
    contours_found = []
    img_copy = img.copy()
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == threshold or threshold == 0:
                x, y, w, h = cv2.boundingRect(approx)
                if draw:
                    cv2.drawContours(img_copy, cnt, -1, color, 3)
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                    cv2.circle(img_copy, (x + (w // 2), y + (h // 2)), 5, color, cv2.FILLED)
                cx, cy = x + (w // 2), y + (h // 2)
                contours_found.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": (cx, cy)})

    contours_found = sorted(contours_found, key=lambda a: a["area"], reverse=True)

    return img_copy, contours_found


def predict(img, pos_x, pos_y, x_max=0, draw=True):
    if pos_x and pos_y:
        a, b, c = np.polyfit(pos_x, pos_y, 2)

        if draw:
            for x in range(x_max + 1):
                y = int(a * (x**2) + b * x + c)
                cv2.circle(img, (x, y), 2, (255, 0, 255), cv2.FILLED)

        c -= Y_BASKET
        x = (-b - math.sqrt(b**2 - 4 * a * c)) // (2 * a)
        return X_RANGE[0] < x < X_RANGE[1]
    return False


def draw_path(img, pos_x, pos_y):
    for i, (x, y) in enumerate(zip(pos_x, pos_y)):
        cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
        if i:
            cv2.line(img, (x, y), (pos_x[i - 1], pos_y[i - 1]), (0, 0, 255), 2)


def draw_result(img, text, position, scale=5, thickness=5, offset=20, text_color=(255, 255, 255),
                rect_color=(0, 200, 0), font=cv2.FONT_HERSHEY_PLAIN):
    x, y = position
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = x - offset, y + offset, x + w + offset, y - h - offset
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

    return img, [x1, y2, x2, y1]


def main():
    # Initialize the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, img = cap.read()

    # Create color finder object
    color_find = ColorFinder(False)

    pos_x = []
    pos_y = []
    x_max = len(img[0])
    res = False
    pause = False
    while success:
        if not pause:
            success, img = cap.read()
            if success:
                img = img[:900, :]

                # Find the color
                img_color, mask = color_find.update(img, BALL_HSV)

                # Find the ball
                img_contours, contours = find_ball(img, mask, min_area=500, draw=False)
                if contours:
                    cx, cy = contours[0]['center']
                    pos_x.append(cx)
                    pos_y.append(cy)

                # Do prediction
                if len(pos_x) <= POINTS:
                    res = predict(img_contours, pos_x, pos_y, x_max, True)

                # Display the image
                if len(pos_x) >= POINTS:
                    text = 'Miss'
                    color = (0, 0, 200)
                    if res:
                        text = 'Basket'
                        color = (0, 200, 0)
                    draw_result(img_contours, text, (50, 150), rect_color=color)
                draw_path(img_contours, pos_x, pos_y)
                img = cv2.resize(img_contours, (0, 0), None, 0.7, 0.7)
                cv2.imshow("ImgColor", img)
                if len(pos_x) == POINTS:
                    pause = True
        key = cv2.waitKey(DELAY)
        if pause and key & 0xFF == ord('s'):
            pause = False
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
