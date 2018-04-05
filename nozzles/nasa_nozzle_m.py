import numpy as np
import matplotlib.pyplot as plt
def nasa_nozzle(t, x):
   mat = np.zeros((3, 3))
   mat[0, :] = [1, -1, 0]
   mat[1, :] = [1, 1, 0]
   mat[2, :] = [1, 1, np.cos((1./x[3] -1 ) * np.pi) - 1]

   a = np.dot(np.linalg.inv(mat), x[:3])

   print(a[0], a[1], a[2])
   s1 = a[0] + a[1] * np.cos(t * np.pi / x[3] - np.pi)
   s2 = a[0] + a[1] + a[2] * (np.cos(t * np.pi / x[3] - np.pi) - 1)

   s = np.append(s1[t < x[3]], s2[t >= x[3]])
   tt = 10 * t
   return np.append(a, [x[3]])
   #return np.sqrt(s / np.pi)

t = np.linspace(0, 1, 100)
x = np.array([0.5, .25, .8, .6])
# x[3] >.5 and < 1
# x[0] and x[1] are > 0 and < a_up
# x[2] <= min(x[0], x[1])
# Generate ten different .msh files that satisfy this :)
# For a mesh file, we want to know which x's go with with it.
print(nasa_nozzle(t, x))
quit()
plt.plot(t, nasa_nozzle(t, x))
plt.show()

