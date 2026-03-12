import numpy as np
import matplotlib.pyplot as plt
import math
import os

array1 = []
array2 = []

with open('moment_rate_kamcatka.txt', 'r') as file:
    for line in file:
        columns = line.strip().split()
        if len(columns) >= 2:
            array1.append(float(columns[0]))  # или int()
            array2.append(float(columns[1]))  # или int()

G = 3 * 10**10
A0 = 1.3 #амплитуда смещений, m
vr = 2000 #скорость распространения, m/s
sigmax = 4500
sigmay = 7000
xx0 = 17000
yy0 = 0

a = 15000 #m
k = 1.13 # = b / a, m
x0 = 5000 #смещение гипоцентра относительно нуля
b = k * a
l = max(a, int(b))

t = np.arange(0, (2 * l - x0) / vr + 1, 0.1) #сетка по времени, s
y_ax = np.arange(0, k*a + 1, 5)#сетка по y, m

dt = t[1] - t[0]
dy = y_ax[1] - y_ax[0]

x_ax = np.arange(0, 2*a + 1, dy)#сетка по x, m

y_max = np.zeros(len(x_ax)) #массив, определяющий границы сетки по y в каждой точке х, согласно параметрам эллипса

for i in range(len(x_ax) - 1):
    N_y = 0
    while N_y * dy <= k * a * (1 - ((x_ax[i + 1] - a) / a) ** 2) ** 0.5:
        N_y += 1
    y_max[i + 1] = N_y * dy

#plt.plot(x_ax, y_max)
#plt.show()

#функция распределения смещения
def spherical(x, y, x0, A0):
    r = ((x - x0)**2 + y**2)**0.5

    # Избегаем деления на ноль в центре
    if r < 0.001:  # маленькое расстояние от центра
        return 1.0

    A = A0 / r

    return A

def const(A):
    return A

def gaussian(x, y, xx0, yy0, A0, sigmax, sigmay):
    A = A0 * math.exp(- 0.5 * (((x - xx0) / sigmax)**2 + ((y - yy0) / sigmay)**2))
    return A

#вводим сетку по углам фи, отсчёт угла от радиус вектора распространения с центром в x0
dfi = dy / (2 * a - x0) #радиан, условие, чтоб в каждую клеточку сетки попали точки фронта
fi = np.arange(0, math.pi, dfi) #радианы

R = [0] * len(t) #фронт
x = np.zeros((len(t), len(fi)))
y = np.zeros((len(t), len(fi))) #координаты точек радиус вектора фронта (от нуля)

for i in range(len(t) - 1):
    R[i] = vr * t[i]
    #print(f'момент времени t{i}')

    for j in range(len(fi)):
        x[i][j] = x0 + R[i] * math.cos(fi[j]) #в i-тый момент времени j-ая точка отсчёта угла
        y[i][j] = R[i] * math.sin(fi[j])
        #print(f'y(t{i})=', y[i][j])

    #plt.plot(fi, y[i], marker='o',linewidth=0.5, label=f't={t[i]}')
#plt.show()

# определяем, в каких ячейках сетки по x и y лежат все точки фронта в i-й момент времени
y_inside = np.zeros((len(t), len(fi)))

#точки y(t, fi), лежащие внутри эллипса
for i in range(len(t) - 1):
    for j in range(len(fi)):
        # Узнаем, в какой ячейке по оси X мы сейчас находимся
        x_index = int(x[i][j] / dy)

        # Проверяем, не ушел ли X за пределы массива y_max, и меньше ли Y, чем граница
        if 0 <= x_index < len(y_max) and y[i][j] <= y_max[x_index]:
            y_inside[i][j] = y[i][j]
        else:
            y_inside[i][j] = -1

    #print(max(y_inside[i]))
   # plt.plot(fi, y_inside[i], marker='o',linewidth=0.5, label=f't={t[i]}')
#plt.show()

N = np.zeros(len(t)) #количество ячеек из сетки x, y, которые пересекает фронт в каждый момент времени
M0_sph = np.zeros(len(t))
M0_const = np.zeros(len(t))
M0_gauss = np.zeros(len(t))
M0_gauss2 = np.zeros(len(t))
M0_gauss3 = np.zeros(len(t))

for i in range(len(t) - 1):
    n = []  # по у
    m = []  # по х

    for j in range(len(fi)):
        if y_inside[i][j] != -1:  # Считаем только те точки, которые не забраковали
            n.append(math.ceil(y_inside[i][j] / dy))
            m.append(math.ceil(x[i][j] / dy))

    pairs = list(zip(n, m))
    s = dict(zip(pairs, range(len(n))))
    N[i] = len(s)

    #sum_A_sph = sum(spherical(m_val * dy, n_val * dy, x0, A0) for (n_val, m_val) in s.keys())
    sum_A_gauss = sum(gaussian(m_val * dy, n_val * dy, xx0, yy0, A0, sigmax, sigmay) for (n_val, m_val) in s.keys())
    #M0_sph[i + 1] = 2 * G * dy ** 2 * vr * sum_A_sph
    M0_gauss[i + 1] = 2 * G * dy ** 2 * vr * sum_A_gauss
#print(N)
M0_const = 2 * G * dy ** 2 * vr * N * const(A0)
print(sum(M0_gauss)*dt)

M0_norm_sph = [m / max(M0_sph) for m in M0_sph]
array2_norm = [m / max(array2) for m in array2]
#plt.plot(t, M0_norm_sph, label = f'max Mr_sph = {(max(M0_sph) / 1e16):.2f}e+16')
#plt.plot(t, M0_const / max(M0_const), label = f'max Mr_const = {(max(M0_const) / 1e16):.2f}e+16')
plt.plot(t, M0_gauss / max(M0_gauss), label = f'max Mr_gauss = {(max(M0_gauss) / 1e18):.5f}e+18')
plt.plot(array1, array2_norm)
plt.xlabel('t, s')
plt.ylabel('d(M0)/dt, Н*м/с')
plt.title('d(M0(t))/dt')
plt.legend()
plt.text(0.78, 0.2, f'a = {a}, vr = {vr}\nk = b/a = {k}, x0 = {x0}',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid()
plt.savefig('Функция источника.png')
plt.show()