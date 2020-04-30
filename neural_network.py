import numpy as np
import matplotlib.pyplot as plt
import seaborn

# x,y座標
X = np.arange(-1.0, 1.0, 0.2) # 要素数10個
Y = np.arange(-1.0, 1.0, 0.2) # 要素数10個

Z = np.zeros((10, 10))  # 出力を格納する10x10グリッド

w_im = np.array([[4.0,4.0],
                 [4.0,4.0]]) #中間層2x2行列
w_mo = np.array([[1.0],
                 [-1.0]])    #出力層2x1行列

# バイアス
b_im = np.array([3.0,-3.0])  # 中間層
b_mo = np.array([0.1])       # 出力層

def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))  # シグモイド関数

def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return u                 # 恒等関数

# グリッドの各マスでニューラルネットワークの演算
for i in range(10):
    for j in range(10):

        # 順伝播
        inp = np.array([X[i], Y[j]])  # 入力層
        mid = middle_layer(inp, w_im, b_im)  # 中間層
        out = output_layer(mid, w_mo, b_mo)  # 出力層

        # グリッドにNNの出力を格納
        print(out)
        Z[i][j] = out[0]

# グリッドの表示
plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()
plt.show()
