from imageio import imread
import colorsys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# m1_ = 'Image_Colorization_Using_Optimization\\resources\\example3_marked.bmp'
# m2_ = 'Image_Colorization_Using_Optimization\\resources\\example3.bmp'
m1_ = 'Image_Colorization_Using_Optimization\\resources\\baby1.bmp'
m2_ = 'Image_Colorization_Using_Optimization\\resources\\baby.bmp'

m1 = imread(m1_)
m2 = imread(m2_)

m1 = np.array(m1, dtype='float64')/255
m2 = np.array(m2, dtype='float64')/255

channel_Y,_,_ = colorsys.rgb_to_yiq(m2[:, :, 0], m2[:, :, 1], m2[:, :, 2])
_,channel_U,channel_V = colorsys.rgb_to_yiq(m1[:, :, 0], m1[:, :, 1], m1[:, :, 2])

m1 = np.dstack((channel_Y, channel_U, channel_V))

W, L, _ = m1.shape
# W = 265, L = 320

# record the coordinates of the scribbles
scr_index = set()
scr_bool = []
# sigmas records the variances of the intensities in a window (9x9) around a pixel
sigmas = []

def xy2idx(x, y):
    return x * L + y

def idx2xy(idx):
    return idx // L, idx % L

def calc_mius_and_sigmas():
    count_scr = 0
    count_all = 0
    for w in range(W):
        for l in range(L):
            if abs(m1[w][l][1]) + abs(m1[w][l][2]) > 1e-4:
                count_scr += 1
                scr_index.add(count_all)
                scr_bool.append(True)
            else:
                scr_bool.append(False)

            nbhd = []
            d = 1
            for i in range(max(w - d, 0), min(w + d + 1, W)):
                for j in range(max(l - d, 0), min(l + d + 1, L)):
                    if i != w or j != l:
                        nbhd.append(m1[i][j][0])

            var = np.var(nbhd)
            sigmas.append(var)
            count_all += 1

    return count_scr

calc_mius_and_sigmas()

M = W * L

A = sparse.lil_matrix((M, M))

def wt_func(sigma, Y_r, Y_s):
    if sigma != 0:
        res = np.exp(-(Y_r - Y_s)**2 / (2 * sigma**2))
        return res if res > 1e-6 else 1e-6
    else:
        return 0

def calc_weights_matrix():
    idx = 0
    for w in range(W):
        for l in range(L):
            if not scr_bool[idx]:
                sigma_r = sigmas[idx]

                # iterate over a window around[w][l]:
                nbhd_w = []
                nbhd_idx = []
                d = 1

                for i in range(max(w - d, 0), min(w + d + 1, W)):
                    for j in range(max(l - d, 0), min(l + d + 1, L)):
                        if i != w or j != l:
                            win_idx = xy2idx(i, j)
                            w_rs = wt_func(sigma_r, m1[w][l][0], m1[i][j][0])

                            d_x = i - w
                            d_y = j - l
                            # diagonal
                            if (d_x == -1 and d_y == 1) or\
                                (d_x == 1 and d_y == 1) or\
                                (d_x == 1 and d_y == -1) or\
                                (d_x == -1 and d_y == -1):
                                w_rs /= np.sqrt(2)

                            nbhd_w.append(w_rs)
                            nbhd_idx.append(win_idx)

                sum_ = sum(nbhd_w)
                if sum_ != 0:
                    nbhd_w /= sum_
                else:
                    len_ = len(nbhd_w)
                    nbhd_w = np.full(len_, 1/len_)

                for _ in range(len(nbhd_w)):
                    A[idx, nbhd_idx[_]] = -nbhd_w[_]
                    
            A[idx, idx] = 1
            idx += 1

calc_weights_matrix()

A = A.tocsc()

b_u = np.zeros(M, dtype='float64')
b_v = np.zeros(M, dtype='float64')

for _ in range(M):
    if _ in scr_index:
        x, y = idx2xy(_)
        b_u[_] = m1[x][y][1]
        b_v[_] = m1[x][y][2]

x_u = spsolve(A, b_u)
x_v = spsolve(A, b_v)

x_u = x_u.reshape((W, L))
x_v = x_v.reshape((W, L))

new = np.zeros((W, L, 3))
new[:, :, 0] = m1[:, :, 0]
new[:, :, 1] = x_u[:, :]
new[:, :, 2] = x_v[:, :]

def yuv_channels_to_rgb(cY,cU,cV):
    ansRGB = [colorsys.yiq_to_rgb(cY[_], cU[_], cV[_]) for _ in range(M)]
    ansRGB = np.array(ansRGB)
    pic_ansRGB = np.zeros((W, L, 3))
    pic_ansRGB[:, :, 0] = ansRGB[:, 0].reshape((W, L))
    pic_ansRGB[:, :, 1] = ansRGB[:, 1].reshape((W, L))
    pic_ansRGB[:, :, 2] = ansRGB[:, 2].reshape((W, L))
    return pic_ansRGB

new = yuv_channels_to_rgb(new[:, :, 0].reshape(M), new[:, :, 1].reshape(M), new[:, :, 2].reshape(M))
imgplot = plt.imshow(new)
plt.savefig('Image_Colorization_Using_Optimization\\python\\Output_ICUO.jpg')
plt.show()
