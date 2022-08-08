import cv2 as cv
import numpy as np
# from paddleocr import PaddleOCR

DEBUG = True
COFF = 0.4
MIN_MATCH_COUNT = 3


def get_icon_location(screen_shot, target):
    global DEBUG, COFF, MIN_MATCH_COUNT
    img1 = cv.imread(screen_shot)
    img2 = cv.imread(target)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=0, trees=10)
    search_params = dict(checks=500)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < COFF * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

        h, w, d = img2.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, m)
        s = dst.shape
        r_dst = dst.reshape(s[0], 2)
        box = []
        for poi in r_dst:
            box.append(poi.astype(np.int32))

        if DEBUG:
            print(box)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.4 * n.distance:
                    matchesMask[i] = [1, 0]

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)

            resultImg1 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

            cv.namedWindow('icon', cv.WINDOW_NORMAL)
            cv.imshow("icon", resultImg1)
            cv.waitKey()

        return np.asarray(box)
    else:
        return None


"""def get_texts(screen_shot):
    global DEBUG
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = ocr.ocr(screen_shot, cls=True)
    if DEBUG:
        for line in result:
            print(line)

    return result"""


def extended_box(icon_box, text_box):
    x = [cor[0] for cor in icon_box] + [cor[0] for cor in text_box]
    y = [cor[1] for cor in icon_box] + [cor[1] for cor in text_box]
    x_min = int(min(x))
    x_max = int(max(x))
    y_min = int(min(y))
    y_max = int(max(y))

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def get_location(screen_shot, target_icon, all_texts=None, target_text=None):
    icon_location = get_icon_location(screen_shot, target_icon)

    if all_texts and target_text:
        icon_related_text_location = []
        for line in all_texts:
            if line[1][0] == target_text:
                icon_related_text_location.extend(line[0])
                break
        return extended_box(icon_location, icon_related_text_location)
    return icon_location
    # if len(icon_related_text_location) > 1:


