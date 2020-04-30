import numpy as np
import matplotlib.pyplot as plt
import seaborn

# x,y座標
X = np.arange(-1.0, 1.0, 0.1) # 要素数20個
Y = np.arange(-1.0, 1.0, 0.1)


w_im = np.array([[1.0,2.0],
                 [2.0,3.0]]) #中間層2x2行列
w_mo = np.array([[-1.0,1.0],
                 [1.0,-1.0]])    #出力層2x2行列

# バイアス
b_im = np.array([0.3,-0.3])  # 中間層
b_mo = np.array([0.4,0.1])   # 出力層


def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))  # シグモイド関数

def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return np.exp(u)/np.sum(np.exp(u))   # ソフトマックス関数

# 分類結果を格納するリスト
x_1 = []
y_1 = []
x_2 = []
y_2 = []


# グリッドの各マスでニューラルネットワークの演算
for i in range(20):
    for j in range(20):

        # 順伝播
        inp = np.array([X[i], Y[j]])  # 入力層
        mid = middle_layer(inp, w_im, b_im)  # 中間層
        out = output_layer(mid, w_mo, b_mo)  # 出力層

        if out[0] > out[1]:
            x_1.append(X[i])
            y_1.append(Y[j])
        else:
            print(out[0], out[1])
            x_2.append(X[i])
            y_2.append(Y[j])

# グリッドの表示
plt.scatter(x_1, y_1, marker="+")
plt.scatter(x_2, y_2, marker="o")
plt.show()
