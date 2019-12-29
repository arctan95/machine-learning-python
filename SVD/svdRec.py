from numpy import *
def loadExdata():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

# 测试
Data = loadExdata()
U, Sigma, VT = linalg.svd(Data)
print(Sigma)
Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
print(U[:,:3] * Sig3 * VT[:3,:])