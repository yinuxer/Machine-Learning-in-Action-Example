#导入numpy模块
from numpy import *
print(random.rand(4, 4))
randMat = mat(random.rand(4, 4)) #array -> matrix
print(randMat.I) #.I是求逆运算

invRandMat = randMat.I
myEye = randMat * invRandMat #计算结果也是4*4的单位矩阵
errormat = myEye - eye(4) #eye(4) 创建4*4的单位矩阵
print(errormat)