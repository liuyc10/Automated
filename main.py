# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

debug = True


def abc():
    path_org = "testpic.jpg"
    path_icon = "ic_power_2x.png"
    img_org = cv.imread(path_org)
    img_icon = cv.imread(path_icon)

    grayA = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)
    # cv.imshow("grayA", grayA)
    grayB = cv.cvtColor(img_icon, cv.COLOR_BGR2GRAY)
    # cv.imshow("grayB", grayB)
    # sift = cv.SIFT_create(float(1000))
    sift = cv.xfeatures2d.SURF_create(float(1000))
    # 寻找关键点和描述符
    keypointsA, featuresA = sift.detectAndCompute(grayA, None)
    keypointsB, featuresB = sift.detectAndCompute(grayB, None)

    # 画出关键点（特征点）

    kpImgA = cv.drawKeypoints(grayA, keypointsA, img_org)
    kpImgB = cv.drawKeypoints(grayB, keypointsB, img_icon)
    # cv.imshow("kpImgA", kpImgA)
    # cv.imshow("kpImgB", kpImgB)

    # FLANN 参数
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # 使用FlannBasedMatcher 寻找最近邻近似匹配
    flann = cv.FlannBasedMatcher(index_params, search_params)
    # 使用knnMatch匹配处理，并返回匹配matches
    matches = flann.knnMatch(featuresA, featuresB, k=2)
    # 通过掩码方式计算有用的点
    matchesMask = [[0, 0] for i in range(len(matches))]
    # 通过coff系数来决定匹配的有效关键点数量。
    coff = 0.2  # 0.1 0.7  0.8  参数可以自己修改进行测试

    # 还是通过描述符的距离进行选择需要的点

    for i, (m, n) in enumerate(matches):

        if m.distance < coff * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    # 使用drawMatchesKnn画出匹配点之间的连线。

    # 分别用灰度图和彩图表达。结果还是满意的。

    resultImg = cv.drawMatchesKnn(grayA, keypointsA, grayB, keypointsB, matches, None, **draw_params)

    resultImg1 = cv.drawMatchesKnn(img_org, keypointsA, img_icon, keypointsB, matches, None, **draw_params)

    plt.imshow(resultImg, ), plt.show()

    cv.imshow("resultImg", resultImg)

    cv.imshow("resultImg1", resultImg1)

    cv.waitKey(0)

    cv.destroyAllWindows()


def defg():
    # 读取图片，以1.png为例
    img1 = cv.imread('testpic.jpg')
    img2 = cv.imread('ic_power_3x.png')
    # 使用SIFT算法获取图像特征的关键点和描述符
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 定义FLANN匹配器
    indexParams = dict(algorithm=0, trees=10)
    searchParams = dict(checks=500)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    # 使用KNN算法实现图像匹配，并对匹配结果排序
    matches = flann.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)

    # 去除错误匹配，0.5是系数，系数大小不同，匹配的结果页不同
    goodMatches_1 = []
    # goodMatches_2 = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            goodMatches_1.append(m)
            # goodMatches_2.append(n)

    # 获取某个点的坐标位置
    # index是获取匹配结果的中位数
    index_1 = int(len(goodMatches_1) / 2)
    # index_2 = int(len(goodMatches_2) / 2)
    # queryIdx是目标图像的描述符索引
    x_1, y_1 = kp1[goodMatches_1[index_1].queryIdx].pt
    x_2, y_2 = kp2[goodMatches_1[index_1].trainIdx].pt
    # 将坐标位置勾画在1.png图片上，并显示
    """cv.rectangle(img1, (int(x_1), int(y_1)), (int(x_1) + 5, int(y_1) + 5), (0, 255, 0), 2)
    cv.rectangle(img2, (int(x_2), int(y_2)), (int(x_2) + 5, int(y_2) + 5), (0, 255, 0), 2)
    cv.namedWindow('1', cv.WINDOW_NORMAL)
    cv.resizeWindow('1', 360, 840)
    cv.imshow('1', img1)
    cv.imshow('2', img2)
    cv.waitKey()"""

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches_1]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches_1]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

    h, w, d = img2.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    x1, y1 = dst[0][0]
    x2, y2 = dst[2][0]
    cv.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.4 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    resultImg1 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(resultImg1, ), plt.show()
    cv.namedWindow('resultImg', cv.WINDOW_NORMAL)
    cv.imshow("resultImg", resultImg1)
    cv.waitKey()


def ocr_t():
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    img_path = 'testpic.jpg'
    result = ocr.ocr(img_path, cls=True)
    for line in result:
        print(line)

    img = Image.open(img_path)
    boxes = [line[0] for line in result]
    texts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(img, boxes, texts, scores, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')


def get_icon_location(screen_shot, target):
    img1 = cv.imread(screen_shot)
    img2 = cv.imread(target)

    # sift = cv.SIFT_create()
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=0, trees=10)
    search_params = dict(checks=500)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

    h, w, d = img2.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    box = [line[0] for line in dst]

    if debug:
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

        x1, y1 = box[0]
        x2, y2 = box[2]

        cv.rectangle(resultImg1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv.namedWindow('icon', cv.WINDOW_NORMAL)
        cv.imshow("icon", resultImg1)
        cv.waitKey()

    return box


def get_texts(screen_shot):
    global debug
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = ocr.ocr(screen_shot, cls=True)
    if debug:
        for line in result:
            print(line)

    return result


def extend_box(icon_box, text_box):
    x = [cor[0] for cor in icon_box] + [cor[0] for cor in text_box]
    y = [cor[1] for cor in icon_box] + [cor[1] for cor in text_box]
    x_min = int(min(x))
    x_max = int(max(x))
    y_min = int(min(y))
    y_max = int(max(y))

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def get_location(screen_shot, target_icon, all_texts, target_text):
    icon_location = get_icon_location(screen_shot, target_icon)
    icon_related_text_location = []
    # icon_related_text_location_list = []

    for line in all_texts:
        if line[1][0] == target_text:
            icon_related_text_location = line[0]

    return extend_box(icon_location, icon_related_text_location)

"""    for line in all_texts:
        if line[1][0] == target_text:
            icon_related_text_location = line[0]
            break
        elif target_text in line[1][0]:
            icon_related_text_location_list.append(line[0])

    if len(icon_related_text_location) != 0:
        return extend_box(icon_location, icon_related_text_location)
    else:
        return None"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    debug = True
    img1 = 'testpic.jpg'
    img2 = 'ic_power_3x.png'
    texts = get_texts(img1)
    rectangle_location = get_location(img1, img2, texts, '设备开关机')

    if debug:
        print(rectangle_location)

    img = cv.imread(img1)
    cv.rectangle(img, rectangle_location[0], rectangle_location[2], (0, 0, 255), 2)
    cv.namedWindow('resultImg', cv.WINDOW_NORMAL)
    cv.imshow("resultImg", img)
    cv.waitKey()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
