import numpy as np

win_count = 0
tie_count = 0
lose_count = 0

def f(file):
    global win_count, tie_count, lose_count
    x = open(file, 'r')
    content = x.readlines()
    win_count += int(content[0].split(':')[1])
    tie_count += int(content[1].split(':')[1])
    lose_count += int(content[2].split(':')[1])


f('result_1.txt')
f('result_2.txt')
f('result_3.txt')
f('result_4.txt')
f('result_5.txt')
f('result_6.txt')

print('win rate:{:.4f}'.format(win_count / (win_count + tie_count + lose_count)))
print('tie rate:{:.4f}'.format(tie_count / (win_count + tie_count + lose_count)))
print('lose rate:{:.4f}'.format(lose_count / (win_count + tie_count + lose_count)))