import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


path_exp_1_pos = "kosar/Gait 1_pos.tsv"

path_exp_1_angle_l_l = "kosar/Gait 1_LL_Angle.tsv"
path_exp_1_angle_vel_l_l = "Gait 1_LL_Angular_Velocity.tsv"
path_exp_1_angle_acc_l_l = "Gait 1_LL_Angular_Acceleration.tsv"

path_exp_1_angle_l_r = "kosar/Gait 1_LR_Angle.tsv"
path_exp_1_angle_vel_l_r = "Gait 1_LR_Angular_Velocity.tsv"
path_exp_1_angle_acc_l_r = "Gait 1_LR_Angular_Acceleration.tsv"

path_exp_1_angle_k_l = "kosar/Gait 1_KL_Angle.tsv"
path_exp_1_angle_vel_k_l = "Gait 1_KL_Angular_Velocity.tsv"
path_exp_1_angle_acc_k_l = "Gait 1_KL_Angular_Acceleration.tsv"

path_exp_1_angle_k_r = "kosar/Gait 1_KR_Angle.tsv"
path_exp_1_angle_vel_k_r = "Gait 1_KR_Angular_Velocity.tsv"
path_exp_1_angle_acc_k_r = "Gait 1_KR_Angular_Acceleration.tsv"
path_exp_1_angle_vel_k_r_pre = "Gait 1_KR_Angular_Velocity_pre.tsv"
path_exp_1_angle_vel_k_r_pre_post = "Gait 1_KR_Angular_Velocity_pre_post.tsv"
path_exp_1_angle_acc_k_r_pre = "Gait 1_KR_Angular_Acceleration_pre.tsv"
path_exp_1_angle_acc_k_r_pre_post = "Gait 1_KR_Angular_Acceleration_pre_post.tsv"

path_exp_1_angle_h_l = "kosar/Gait 1_HL_Angle.tsv"
path_exp_1_angle_vel_h_l = "Gait 1_HL_Angular_Velocity.tsv"
path_exp_1_angle_acc_h_l = "Gait 1_HL_Angular_Acceleration.tsv"

path_exp_1_angle_h_r = "kosar/Gait 1_HR_Angle.tsv"
path_exp_1_angle_vel_h_r = "Gait 1_HR_Angular_Velocity.tsv"
path_exp_1_angle_acc_h_r = "Gait 1_HR_Angular_Acceleration.tsv"


def read_data(filename):
    time = []
    res_angle = []

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i < 8:
                continue
            try:
                frame, t, angle = line.strip().split('\t')
                time.append(float(t))
                res_angle.append(float(angle))
            except ValueError:
                print(f"Warning: Skipped line {i+1} due to parsing error: {line.strip()}")
                continue
    return np.array(time), np.array(res_angle)


def read_data_pos(filename):
    time = []
    data = []

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i < 8:
                continue
            try:
                parts = line.split('\t')
                frame = int(parts[0])
                t = float(parts[1])
                positions = [float(x) for x in parts[2:]]

                if len(positions) != 30:
                    print(
                        f"Warning: Line {i + 1} has incorrect number of data points. Expected {10 * 3}, got {len(positions)}. Skipping.")
                    continue

                time.append(t)
                positions_array = np.array(positions).reshape(10, 3)
                data.append(positions_array)

            except ValueError as e:
                print(f"Warning: Skipped line {i + 1} due to parsing error: {line}. Error: {e}")
                continue
            except IndexError as e:
                print(f"Warning: Skipped line {i + 1} due to index error: {line}. Error: {e}")
                continue

    return np.array(time), np.array(data)


def animate_3d_scatter(time, data, output_filename):
    num_frames, num_points, _ = data.shape  # _ = 3, координаты x, y, z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatters = [ax.scatter([], [], [], marker='o') for i in range(num_points)]

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Motion Capture Animation")

    x_min = np.min(data[:, :, 0])
    x_max = np.max(data[:, :, 0])
    y_min = np.min(data[:, :, 1])
    y_max = np.max(data[:, :, 1])
    z_min = np.min(data[:, :, 2])
    z_max = np.max(data[:, :, 2])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.view_init(elev=20, azim=45)  # Наклон и угол обзора

    # текстовое поле для отображения времени/кадра
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)  # Позиция в верхнем левом углу

    def update(frame):
        for i, scatter in enumerate(scatters):
            x = data[frame, i, 0]
            y = data[frame, i, 1]
            z = data[frame, i, 2]
            scatter._offsets3d = ([x], [y], [z])

        time_text.set_text(f"Time: {time[frame]:.3f} s, Frame: {frame + 1}")  # Форматируем вывод времени
        return scatters + [time_text]  # Важно вернуть список обновленных объектов, включая time_text

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False, repeat=True)  # blit=False важно для 3D
    ani.save(output_filename, writer='ffmpeg', fps=30)  # writer='imagemagick'

def fit_polynomials(dim, phi1, time1, phi2, time2):
    if not np.array_equal(time1, time2):
        phi2 = np.interp(time1, time2, phi2)

    # Создаем список столбцов матрицы A
    columns = [phi1 ** i for i in range(dim + 1)]
    # Создаем матрицу A для метода наименьших квадратов (phi1^0, phi1^1, phi1^2, phi1^3, ...)
    A = np.vstack(columns).T
    # A * coeffs = phi2
    coeffs, _, _, _ = np.linalg.lstsq(A, phi2, rcond=None)
    return coeffs

if __name__ == '__main__':
    frame_sit_1_start = 200 # 200
    frame_sit_1_end = 270 # 315
    frame_stand_1_start = 400
    frame_stand_1_end = 515

    # time, data = read_data_pos(path_exp_1_pos)
    # animate_3d_scatter(time, data, "anim/animation_1.mp4")

    time_hr, hr_angle = read_data(path_exp_1_angle_h_l)
    time_kr, kr_angle = read_data(path_exp_1_angle_k_l)
    time_lr, lr_angle = read_data(path_exp_1_angle_l_l)

    time_hl, hl_angle = read_data(path_exp_1_angle_h_l)
    time_kl, kl_angle = read_data(path_exp_1_angle_k_l)
    time_ll, ll_angle = read_data(path_exp_1_angle_l_l)

    # fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # 3 строки, 1 столбец
    # axes[0].plot(time_hr, hr_angle)
    # axes[0].set_xlabel("Время (секунды)")
    # axes[0].set_ylabel("HR_Angle (градусы)")
    # axes[0].set_title("Зависимость HR_Angle от времени")
    # axes[0].grid(True)
    # axes[1].plot(time_kr, kr_angle)
    # axes[1].set_xlabel("Время (секунды)")
    # axes[1].set_ylabel("KR_Angle (градусы)")
    # axes[1].set_title("Зависимость KR_Angle от времени")
    # axes[1].grid(True)
    # axes[2].plot(time_lr, lr_angle)
    # axes[2].set_xlabel("Время (секунды)")
    # axes[2].set_ylabel("LR_Angle (градусы)")
    # axes[2].set_title("Зависимость LR_Angle от времени")
    # axes[2].grid(True)
    # plt.tight_layout() # Автоматическая подгонка расположения графиков, чтобы избежать наложений
    # plt.show()

    time_hr = time_hr[frame_sit_1_start:frame_sit_1_end]
    hr_angle = hr_angle[frame_sit_1_start:frame_sit_1_end]
    time_kr = time_kr[frame_sit_1_start:frame_sit_1_end]
    kr_angle = kr_angle[frame_sit_1_start:frame_sit_1_end]
    time_lr = time_lr[frame_sit_1_start:frame_sit_1_end]
    lr_angle = lr_angle[frame_sit_1_start:frame_sit_1_end]

    # fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # 3 строки, 1 столбец
    # axes[0].plot(time_hr, hr_angle)
    # axes[0].set_xlabel("Время (секунды)")
    # axes[0].set_ylabel("HR_Angle (градусы)")
    # axes[0].set_title("Зависимость HR_Angle от времени")
    # axes[0].grid(True)
    # axes[1].plot(time_kr, kr_angle)
    # axes[1].set_xlabel("Время (секунды)")
    # axes[1].set_ylabel("KR_Angle (градусы)")
    # axes[1].set_title("Зависимость KR_Angle от времени")
    # axes[1].grid(True)
    # axes[2].plot(time_lr, lr_angle)
    # axes[2].set_xlabel("Время (секунды)")
    # axes[2].set_ylabel("LR_Angle (градусы)")
    # axes[2].set_title("Зависимость LR_Angle от времени")
    # axes[2].grid(True)
    # plt.tight_layout()
    # plt.show()

    # fig, axes = plt.subplots(2, 1, figsize=(8, 12))  # 3 строки, 1 столбец
    # axes[0].plot(lr_angle, hr_angle)
    # axes[0].set_xlabel("Время (секунды)")
    # axes[0].set_ylabel("HR_Angle (lr_angle)")
    # axes[0].set_title("Зависимость HR_Angle от lr_angle")
    # axes[0].grid(True)
    # axes[1].plot(lr_angle, kr_angle)
    # axes[1].set_xlabel("Время (секунды)")
    # axes[1].set_ylabel("KR_Angle (lr_angle)")
    # axes[1].set_title("Зависимость KR_Angle от lr_angle")
    # axes[1].grid(True)
    # plt.tight_layout()
    # plt.show()

    lambda_coeffs = fit_polynomials(3, lr_angle, time_lr, kr_angle, time_kr)
    betta_coeffs = fit_polynomials(3, lr_angle, time_lr, hr_angle, time_hr)

    phi2_approx = np.polyval(lambda_coeffs[::-1], lr_angle)  # Переворачиваем порядок коэффициентов для polyval
    phi3_approx = np.polyval(betta_coeffs[::-1], lr_angle)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(lr_angle, kr_angle, label="phi2 (оригинал)")
    plt.plot(lr_angle, phi2_approx, label="phi2 (аппроксимация)")
    plt.xlabel("lr_angle")
    plt.ylabel("phi2")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(lr_angle, hr_angle, label="phi3 (оригинал)")
    plt.plot(lr_angle, phi3_approx, label="phi3 (аппроксимация)")
    plt.xlabel("lr_angle")
    plt.ylabel("phi3")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()