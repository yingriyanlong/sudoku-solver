# !/usr/bin/python
# coding:utf8
# author:ykf
# latest date:2017/1/28
# environment:python3.6.0 + opencv 3.2 with contributes + pytesseract

import time
import sudoku_solver
import sudoku_recognizer

file_dir = input(u'请输入要解的数独题目照片:')
start = time.clock()
image_origin = sudoku_recognizer.image_reader(file_dir)
image_segment = sudoku_recognizer.sudoku_segment(image_origin)
numbers = sudoku_recognizer.numbers_segment(image_segment)
answer, iteration = sudoku_solver.solver(numbers)
stop = time.clock()
print(u'----------原题-----------')
for i in range(9):
    for j in range(9):
        print(numbers[i * 9 + j], end=u'　')
    print('\n', end='')
print(u'----------答案-----------')
if answer:
    for i in range(9):
        for j in range(9):
            print(answer[i * 9 + j], end=u'　')
        print('\n', end='')
else:
    print('recognizer error')
print('-------------------------')

print(str(stop - start).join(u'共使用时间： s'.split()))
print(str(iteration).join(u'求解循环次数： 次'.split()))
