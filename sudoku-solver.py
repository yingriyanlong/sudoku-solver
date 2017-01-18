# !/usr/bin/python
# coding:utf8
# author:ykf
# latest date:2017/1/19
# environment:python3.6

# import python libs
# import time
import copy


# import string


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
                same_row = range(row * 9, (row + 1) * 9)  # 同行元素位置
                same_col = range(col, 81, 9)  # 同列元素位置
                # 同方块元素位置
                same_block = (list(range(block_row * 27 + block_col * 3, block_row * 27 + (block_col + 1) * 3)) +
                              list(range(block_row * 27 + 9 + block_col * 3, block_row * 27 + 9 + (block_col + 1) * 3)) +
                              list(range(block_row * 27 + 18 + block_col * 3, block_row * 27 + 18 + (block_col + 1) * 3)))
                # 题目没有被解决，标志位置否
                is_solved = False
                # 当前位置可供选择的数字
                sd_current_available_number = list(range(1, 10))
                # 删除行内出现的元素
                for i in same_row:
                    if sd_current_available_number.__contains__(sd_current_solution[i]):
                        sd_current_available_number.remove(sd_current_solution[i])
                # 删除列内出现的元素
                for i in same_col:
                    if sd_current_available_number.__contains__(sd_current_solution[i]):
                        sd_current_available_number.remove(sd_current_solution[i])
                # 删除同方块内的元素
                for i in same_block:
                    if sd_current_available_number.__contains__(sd_current_solution[i]):
                        sd_current_available_number.remove(sd_current_solution[i])
                # 若当前位置可选数字为最小 ，替换坐标和选择数字
                if sd_current_available_number.__len__() < sd_min_available_number.__len__():
                    sd_min_available_number_x = X
                    sd_min_available_number = copy.deepcopy(sd_current_available_number)
                # 若最小位置的可供选择数字数目为0，退出循环
                if sd_min_available_number.__len__() == 0:
                    break

        for number in sd_min_available_number:
            sd_new_solution = copy.deepcopy(sd_current_solution)
            sd_new_solution[sd_min_available_number_x] = number
            sdtemp.append(sd_new_solution)

    if is_solved:
        return sd_current_solution
    else:
        return []


# main函数
if __name__ == "__main__":
    sudoku = [7, 8, 0, 0, 0, 0, 0, 4, 5,
              1, 3, 0, 0, 0, 0, 0, 9, 2,
              0, 0, 6, 0, 0, 0, 7, 0, 0,
              3, 5, 0, 6, 0, 4, 0, 2, 7,
              9, 0, 0, 0, 2, 0, 0, 0, 3,
              8, 7, 0, 1, 0, 3, 0, 6, 9,
              0, 0, 3, 0, 0, 0, 5, 0, 0,
              5, 2, 0, 0, 0, 0, 0, 1, 8,
              6, 9, 0, 0, 0, 0, 0, 3, 4]
    answer = solver(sudoku)
    if answer:
        for i in range(9):
            for j in range(9):
                print(answer[1], end=' ')
            print('\n', end='')
    else:
        print('no solution!')
