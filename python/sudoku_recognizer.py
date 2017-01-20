# !/usr/bin/python
# coding:utf8
# author:ykf
# latest date:2017/1/20
# environment:python3.6
# opencv 3.2 with contributes

# 导入python数据库/import python libs
# from matplotlib import pylab as lb
import cv2
import numpy


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
    image_sudoku_height, image_sudoku_width = image_sudoku_origin.shape[:2]
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
    #     cv2.line(image_sudoku_origin, (rectangle_max[i % 4][0][0], rectangle_max[i % 4][0][1]),
    #              (rectangle_max[(i + 1) % 4][0][0], rectangle_max[(i + 1) % 4][0][1]), (0, 0, 255), 2)
    # cv2.imshow('1', image_sudoku_origin)
    # cv2.waitKey(0)


# main函数
if __name__ == "__main__":
    origin = image_reader(r'C:\1.jpg')
    sudoku_segment(origin)
