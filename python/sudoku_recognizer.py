# !/usr/bin/python
# coding:utf8
# author:ykf
# latest date:2017/1/25
# environment:python3.6
# opencv 3.2 with contributes

# 导入python数据库/import python libs
# from matplotlib import pylab as lb
import cv2
import math
import numpy

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
SUDOKU_SIZE = 9
N_MIN_ACTIVE_PIXELS = 100
# 初始化数独数字分格存放数组
sudoku_numbers = numpy.zeros(shape=(SUDOKU_SIZE * SUDOKU_SIZE, IMAGE_WIDTH * IMAGE_HEIGHT))


# 排序角坐标
def get_sort_rectangle_max(corners):
    ar = [corners[0, 0, :], corners[1, 0, :], corners[2, 0, :], corners[3, 0, :]]

    x_ave = sum(corners[x, 0, 0] for x in range(corners.__len__())) / corners.__len__()
    y_ave = sum(corners[x, 0, 1] for x in range(corners.__len__())) / corners.__len__()

    def algo(v):  # atan((x-avx)/(y-avy))
        return math.atan2(v[0] - x_ave, v[1] - y_ave)

    ar.sort(key=algo)
    return ar[3], ar[0], ar[1], ar[2]


def image_reader(direction):
    # 保证输入的图片路径不为空
    if direction is not None:
        # 使用cv2.imread函数以灰度格式读取图片内容并返回
        image_sudoku_origin = cv2.imread(direction)
        # image_sudoku_origin = cv2.imread(direction, cv2.IMREAD_GRAYSCALE)
        # 若打开的图片为空，表示该地址并没有这个图片
        if image_sudoku_origin is None:
            print("No picture is here.")
        return image_sudoku_origin
    else:
        # 若路径为空，报错并返回空集
        print("Not a correct direction.")
        return None


def sudoku_segment(image_sudoku_origin):
    # 将源图片由RGB模式转换为灰度模式
    image_sudoku_gray = cv2.cvtColor(image_sudoku_origin, cv2.COLOR_BGR2GRAY)
    # image_sudoku_gray = image_sudoku_origin
    # 自适应阀值化，将灰度图片转换为黑白图片
    image_sudoku_binary = cv2.adaptiveThreshold(image_sudoku_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 15, 10)
    # cv2.imshow('2', image_sudoku_binary)
    # 找到图形的轮廓
    _, contours, _ = cv2.findContours(image_sudoku_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 获取图像的尺寸（高height，宽width）
    # image_sudoku_height, image_sudoku_width = image_sudoku_origin.shape[:2]
    # 寻找最大的长方形
    size_rectangle_max = 0
    rectangle_max = []
    for i, c in enumerate(contours):
        # 找近似的多边形
        approximation = cv2.approxPolyDP(c, 8, True)
        # 近似多边形是不是四边形？
        if len(approximation) != 4:
            continue
        # 近似多边形是不是凸的
        if not cv2.isContourConvex(approximation):
            continue
        # 计算近似多边形的面积
        size_rectangle = cv2.contourArea(approximation)
        if size_rectangle > size_rectangle_max:
            size_rectangle_max = size_rectangle
            rectangle_max = approximation
    # 在原图中画出数独题目区域
    # for i, c in enumerate(rectangle_max):
    #     cv2.line(image_sudoku_origin, (rectangle_max[i][0][0], rectangle_max[i][0][1]),
    #              (rectangle_max[(i + 1)%4][0][0], rectangle_max[(i + 1)%4][0][1]), (0, 0, 255), 2)
    #     cv2.line(image_sudoku_origin,(0,0),(0,500),(255,0,0),thickness=5)
    # cv2.imshow('3', image_sudoku_origin)
    # cv2.waitKey(0)

    # 框起来的图片变为576*576的map，每个格子为64*64
    aim_points = numpy.array([numpy.array([0.0, 0.0], numpy.float32) +
                              numpy.array([SUDOKU_SIZE * IMAGE_HEIGHT, 0.0], numpy.float32),
                              numpy.array([0.0, 0.0], numpy.float32),
                              numpy.array([0.0, 0.0], numpy.float32) +
                              numpy.array([0.0, SUDOKU_SIZE * IMAGE_HEIGHT], numpy.float32),
                              numpy.array([0.0, 0.0], numpy.float32) +
                              numpy.array([SUDOKU_SIZE * IMAGE_HEIGHT, SUDOKU_SIZE * IMAGE_WIDTH], numpy.float32),
                              ], numpy.float32)
    sort_rectangle_max = get_sort_rectangle_max(rectangle_max)
    need_to_correct_points = numpy.array(sort_rectangle_max, numpy.float32)
    # 生成转换坐标需要的3*3参数
    pers = cv2.getPerspectiveTransform(need_to_correct_points, aim_points)
    # 重新生成只有数独题目的图片
    warp = cv2.warpPerspective(image_sudoku_binary, pers, (SUDOKU_SIZE * IMAGE_HEIGHT, SUDOKU_SIZE * IMAGE_WIDTH))
    # warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('1', warp)
    # cv2.waitKey(0)
    return warp


# TODO a subprogram to segment numbers into arrays
def numbers_segment(image_sudoku_segment):
    for row in range(SUDOKU_SIZE):
        for col in range(SUDOKU_SIZE):
            # 将第row行第col列的数字的灰度(二进制）图像存入image_number,row<=9,col<=9
            image_number = image_sudoku_segment[row * IMAGE_HEIGHT:(row + 1) * IMAGE_HEIGHT][
                           :, col * IMAGE_WIDTH:(col + 1) * IMAGE_WIDTH]
            # 使用不同的自适阀值参数将灰度转为二进制
            # image_number_threshold = cv2.adaptiveThreshold(image_number, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                                cv2.THRESH_BINARY_INV, 15, 10)
            # 为了减少对数字识别的影响，将距离中心的一定范围内外的白色像素点变为黑色
            image_number_zero = numpy.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH))
            image_number_zero[IMAGE_HEIGHT // 5:IMAGE_HEIGHT - IMAGE_HEIGHT // 5]\
            [:, IMAGE_WIDTH // 5:IMAGE_WIDTH - IMAGE_WIDTH // 5] = image_number\
            [IMAGE_HEIGHT // 5:IMAGE_HEIGHT - IMAGE_HEIGHT // 5][:, IMAGE_WIDTH // 5:IMAGE_WIDTH - IMAGE_WIDTH // 5]
            # 计算白色像素个数
            image_number_white_number = cv2.countNonZero(image_number_zero)
            if image_number_white_number >= N_MIN_ACTIVE_PIXELS:
                pass
            #     cv2.imshow('1', image_number_zero)
            #     cv2.waitKey(0)
            # TODO add next program段: 每个数字的最小框


# main函数
if __name__ == "__main__":
    image_origin = image_reader(r'C:\1.jpg')
    image_segment = sudoku_segment(image_origin)
    numbers_segment(image_segment)
