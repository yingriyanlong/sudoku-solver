# !/usr/bin/python
# coding:utf8
# author:ykf
# latest date:2017/1/19
# environment:python3.6

# import python libs
# import string
import time
import copy
from timeit import Timer

# 解数独函数
def solver(sdin):
    # 将输入的数独题目存入sdtemp用于解题目
    sdtemp = [sdin]
    # iterations为循环次数计数器
    iterations = 0

    while sdtemp.__len__() > 0:
        # 选取最后一个座位当前解
        sd_current_solution = sdtemp.pop()
        # 循环次数加一
        iterations += 1
        # 最少选择处的可供选择的数字
        sd_min_available_number = list(range(1, 10))
        # 最少选择处的坐标
        sd_min_available_number_x = 0
        # 题目是否已解决的标志位
        is_solved = True
        for X in range(81):
            if sd_current_solution[X] == 0:
                row = X // 9  # 当前位置的行
                col = X % 9  # 当前位置的列
                block_row = row // 3  # 当前位置所处方块行
                block_col = col // 3  # 当前位置所处方块列
                same_row = set(range(row * 9, (row + 1) * 9))  # 同行元素位置
                same_col = set(range(col, 81, 9))  # 同列元素位置
                # 同方块元素位置
                same_block = (set(range(block_row * 27 + block_col * 3, block_row * 27 + (block_col + 1) * 3)) |
                              set(range(block_row * 27 + 9 + block_col * 3, block_row * 27 + 9 + (block_col + 1) * 3)) |
                              set(range(block_row * 27 + 18 + block_col * 3, block_row * 27 + 18 + (block_col + 1) * 3)))
                same = same_row | same_col | same_block
                same.remove(X)
                # 题目没有被解决，标志位置否
                is_solved = False
                # 当前位置可供选择的数字
                sd_current_available_number = list(range(1, 10))
                # 删除出现的元素
                for i in same:
                    if sd_current_available_number.__contains__(sd_current_solution[i]):
                        sd_current_available_number.remove(sd_current_solution[i])
                # 若最小位置的可供选择数字数目为0，退出循环
                if sd_min_available_number.__len__() == 0:
                    break
                # 若当前位置可选数字为最小 ，替换坐标和选择数字
                if sd_current_available_number.__len__() < sd_min_available_number.__len__():
                    sd_min_available_number_x = X
                    sd_min_available_number = copy.deepcopy(sd_current_available_number)

        if (sd_min_available_number.__len__() > 0) & (not is_solved):
            for number in sd_min_available_number:
                sd_current_solution[sd_min_available_number_x] = number
                sdtemp.append(sd_current_solution)

    if is_solved:
        return sd_current_solution, iterations
    else:
        return [], iterations


# main函数
if __name__ == "__main__":
    sudoku = [6, 0, 5, 3, 0, 4, 0, 0, 8,
              0, 0, 0, 7, 0, 6, 9, 0, 4,
              0, 4, 7, 5, 0, 0, 0, 3, 1,
              0, 0, 6, 8, 0, 9, 4, 1, 0,
              0, 8, 4, 0, 0, 0, 0, 0, 0,
              5, 0, 3, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 3, 1, 0, 0,
              4, 0, 1, 0, 0, 0, 0, 0, 2,
              0, 0, 9, 2, 0, 0, 8, 7, 5]
    start = time.clock()
    answer, iteration = solver(sudoku)
    stop = time.clock()

    # 输出结果
    if answer:
        for i in range(9):
            for j in range(9):
                print(answer[i*9+j], end=' ')
            print('\n', end='')
    else:
        print('no solution!')
    print("循环次数:", iteration)
    # 使用timeit模块测试时间
    t = Timer("solver","from __main__ import solver")
    print("Timeit时间：",t.timeit(iteration))
    print("TimeClock时间：",stop - start)
