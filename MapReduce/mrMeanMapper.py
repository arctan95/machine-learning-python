#coding=utf-8
# 分布式均值和方差计算的mapper
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = power(input, 2) #对所有的值进行平方

print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)))
sys.stderr.write("report: still alive\n")

