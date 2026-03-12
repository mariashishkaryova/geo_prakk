import numpy as np
import matplotlib.pyplot as plt
import math
import os

path = r'D:\pythonProject\Geo_prak\geo_prak'
OUTPUT_DIR_graph = os.path.join(path, 'theory', 'graph')
os.makedirs(OUTPUT_DIR_graph, exist_ok=True)
data_path = os.path.join(path, 'theory')

for s in range(0, 10):
    G = 3 * 10 ** 11
    A0 = np.random.uniform(0.5, 2) #амплитуда смещений, m
    vr = np.random.uniform(1500, 2500) #скорость распространения, m/s

    a = np.random.uniform(3000, 18000)  # m
    k = np.random.uniform(0.6, 1.3)  # = b / a, m
    b = k * a

    sigmax = np.random.uniform(4000, 5000) #ширина гауссового распределения, м
    sigmay = np.random.uniform(4000, 5000)
    l = max(a, int(b))

    f = np.random.uniform(0, 1)
    x0 = f * a  # смещение гипоцентра относительно нуля, m

    g = np.random.uniform(0.7, 1.3)
    xx0 = g * a #точка максимума гауссового распределения, м
    yy0 = 0

    t = np.arange(0, (2 * l - x0) / vr + 1, 0.1)  # сетка по времени, s
    y_ax = np.arange(0, k * a + 1, 5)  # сетка по y, m

    dt = t[1] - t[0]
    dy = y_ax[1] - y_ax[0]

    x_ax = np.arange(0, 2 * a + 1, dy)  # сетка по x, m

    y_max = np.zeros(len(x_ax))  # массив, определяющий границы сетки по y в каждой точке х, согласно параметрам эллипса

    for i in range(len(x_ax) - 1):
        N_y = 0
        while N_y * dy <= k * a * (1 - ((x_ax[i + 1] - a) / a) ** 2) ** 0.5:
            N_y += 1
        y_max[i + 1] = N_y * dy


    def gaussian(x, y, xx0, yy0, A0, sigmax, sigmay):
        A = A0 * math.exp(- 0.5 * (((x - xx0) / sigmax) ** 2 + ((y - yy0) / sigmay) ** 2))
        return A


    # вводим сетку по углам фи, отсчёт угла от радиус вектора распространения с центром в x0
    dfi = dy / (2 * a - x0)  # радиан, условие, чтоб в каждую клеточку сетки попали точки фронта
    fi = np.arange(0, math.pi, dfi)  # радианы

    R = [0] * len(t)  # фронт
    x = np.zeros((len(t), len(fi)))
    y = np.zeros((len(t), len(fi)))  # координаты точек радиус вектора фронта (от нуля)

    for i in range(len(t) - 1):
        R[i] = vr * t[i]
        # print(f'момент времени t{i}')

        for j in range(len(fi)):
            x[i][j] = x0 + R[i] * math.cos(fi[j])  # в i-тый момент времени j-ая точка отсчёта угла
            y[i][j] = R[i] * math.sin(fi[j])

    # определяем, в каких ячейках сетки по x и y лежат все точки фронта в i-й момент времени
    y_inside = np.zeros((len(t), len(fi)))

    # точки y(t, fi), лежащие внутри эллипса
    for i in range(len(t) - 1):
        for j in range(len(fi)):
            # Узнаем, в какой ячейке по оси X мы сейчас находимся
            x_index = int(x[i][j] / dy)

            # Проверяем, не ушел ли X за пределы массива y_max, и меньше ли Y, чем граница
            if 0 <= x_index < len(y_max) and y[i][j] <= y_max[x_index]:
                y_inside[i][j] = y[i][j]
            else:
                y_inside[i][j] = -1


    M0_gauss = np.zeros(len(t))

    for i in range(len(t) - 1):
        n = []  # по у
        m = []  # по х

        for j in range(len(fi)):
            if y_inside[i][j] != -1:  # Считаем только те точки, которые не забраковали
                n.append(math.ceil(y_inside[i][j] / dy))
                m.append(math.ceil(x[i][j] / dy))

        pairs = list(zip(n, m))
        ss = dict(zip(pairs, range(len(n))))

        sum_A_gauss = sum(gaussian(m_val * dy, n_val * dy, xx0, yy0, A0, sigmax, sigmay) for (n_val, m_val) in ss.keys())
        M0_gauss[i + 1] = 2 * G * dy ** 2 * vr * sum_A_gauss


    M0_sum = sum(M0_gauss)*dt

    plt.figure()
    plt.plot(t, M0_gauss / max(M0_gauss), label=f'max Mr_gauss = {(max(M0_gauss) / 1e16):.5f}e+16')
    plt.xlabel('t, s')
    plt.ylabel('d(M0)/dt, Н*м/с')
    plt.title('d(M0(t))/dt')
    plt.legend()
    plt.text(0.78, 0.2, f'a = {a:.2f}, vr = {vr:.2f}\nk = b/a = {k:.2f}, x0 = {x0:.2f}\nsigma_x = {sigmax:.2f}, Xmax = {xx0:.2f}',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR_graph, f'Функция источника_{s}.png'))
    plt.close()

    with open(os.path.join(data_path, f'data_{s}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"a = {a} м\n")
        f.write(f'b = {b} м\n')
        f.write(f"x0 = {x0} м\n")
        f.write(f"vr = {vr} м/с\n")
        f.write(f"A0 = {A0} м\n")
        f.write(f'Xmax = {xx0} м\n')
        f.write(f'sigmaX = {sigmax} м\n')
        f.write(f'sigmaY = {sigmay} м\n')
        f.write(f'M0 = {M0_sum} Н*м - интегральное значение сейсмического момента\n')
        f.write(f'Mr_max = {max(M0_gauss)} Н*м/с - максимальное значение функции источника\n')
        f.write(f'1-ый столбец - отнормированная функция источника\n')
        f.write(f'2-ой столбец - время\n')
        for a, b in zip(M0_gauss / max(M0_gauss), t):
            f.write(f"{a}\t{b}\n")  # \t - табуляция между столбцами
        f.close()


#после выполнения работы в терминале git bash:
#cd D:\pythonProject\Geo_prak\geo_prak
#git add theory/
#git commit -m "Add generated theory data and graphs"
#git push