import numpy as np
import matplotlib.pyplot as plt
import seaborn

# x,y座標
X = np.arange(-1.0, 1.0, 0.2) # 要素数10個
Y = np.arange(-1.0, 1.0, 0.2) # 要素数10個

Z = np.zeros((10, 10))  # 出力を格納する10x10グリッド

w_x = 2.5 #x,y座標の入力の重み
w_y = 3.0

bias = 0.1 # バイアス

# グリッドの各マスでのニューロンの演算
for i in range(10):
    for j in range(10):
        # 入力と重みの積の総和 + バイアス
        u = X[i] * w_x + Y[j] * w_y + bias

        # グリッドに出力を格納
        y = 1/(1 + np.exp(-u)) # sigmoid function
        Z[i][j] = y

# グリッドの表示
plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()
plt.show()
