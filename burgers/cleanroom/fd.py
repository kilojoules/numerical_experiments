import numba as nb
import numpy as np
@nb.jit
def d1(arr, dnu):
     outv = np.zeros(arr.size)
     outv[0] = (arr[1] - arr[0]) / dnu
     outv[1] = (arr[2] - arr[0]) / (2 * dnu)
     outv[2:-2] = (-1 * arr[4:] + 8 * arr[3:-1] - 8 * arr[1:-3] + arr[:-4]) / (12 * dnu)
     outv[-1] = (arr[-1] - arr[-2]) / dnu
     outv[-2] = (arr[-1] - arr[-3]) / (2 * dnu)
     return outv
@nb.jit
def d2(arr, dnu):
     outv = np.zeros(arr.size)
     outv[0] = (2 * arr[0] - 5 * arr[1] + 4 * arr[2] - arr[3]) / dnu ** 2
     outv[1] = (2 * arr[1] - 5 * arr[2] + 4 * arr[3] - arr[4]) / dnu ** 2
     outv[2:-2] = (-1 * arr[:-4] + 16 * arr[1:-3] - 30 * arr[2:-2] + 16 * arr[3:-1] - arr[4:]) / (12 * dnu ** 2)
     outv[-1] = (2 * arr[-1] - 5 * arr[-2] + 4 * arr[-3] - arr[-4]) / dnu ** 2
     outv[-2] = (2 * arr[-2] - 5 * arr[-3] + 4 * arr[-4] - arr[-5]) / dnu ** 2
     return outv


