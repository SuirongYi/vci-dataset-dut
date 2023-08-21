from math import cos, sin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict
import csv
import os
import pandas as pd

width = 1.50
length = 3.70


def plot_vehicle(x, y, phi):
    x_a = x + width / 2 * sin(phi) - length / 2 * cos(phi)
    y_a = y - width / 2 * cos(phi) - length / 2 * sin(phi)
    x_b = x_a + length * cos(phi)
    y_b = y_a + length * sin(phi)
    x_c = x_a - width * sin(phi)
    y_c = y_a + width * cos(phi)
    x_d = x_a + length * cos(phi) - width * sin(phi)
    y_d = y_a + length * sin(phi) + width * cos(phi)
    ax.plot([x_a, x_b], [y_a, y_b], color='blue', zorder=3)
    ax.plot([x_a, x_c], [y_a, y_c], color='blue', zorder=3)
    ax.plot([x_d, x_b], [y_d, y_b], color='blue', zorder=3)
    ax.plot([x_d, x_c], [y_d, y_c], color='blue', zorder=3)


def get_ped_data(data_path: str, ) -> Dict:
    csv_file = os.path.join(path, data_path)
    data = pd.read_csv(csv_file)
    colum_data = data['frame'].tolist()
    frame_max = max(colum_data)
    temp = {i: [] for i in range(1, frame_max + 1)}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['label'] == 'ped':
                temp[int(row['frame'])].append([float(row['x_est']), float(row['y_est'])])
            else:
                temp[int(row['frame'])].append([float(row['x_est']), float(row['y_est']), float(row['psi_est'])])
    return temp


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)
    ped_csv_file = 'data\\trajectories_filtered\\intersection_05_traj_ped_filtered.csv'
    veh_csv_file = 'data\\trajectories_filtered\\intersection_05_traj_veh_filtered.csv'
    ped_data = get_ped_data(ped_csv_file)
    veh_data = get_ped_data(veh_csv_file)
    fig = plt.figure(figsize=(10, 8))
    plt.ion()
    ax = fig.add_subplot()
    for i in range(1, len(ped_data) + 1):
        plt.cla()
        for ped in ped_data[i]:
            circle = patches.Circle(xy=(ped[0], ped[1]), radius=0.20, fc='red', ec='red', zorder=3)
            ax.add_patch(circle)
        if i in list(veh_data.keys()):
            for veh in veh_data.get(i):
                plot_vehicle(veh[0], veh[1], veh[2])
        ax.axis('equal')
        ax.set_xlim(-5.0, 35.0)
        ax.set_ylim(-5.0, 35.0)
        plt.draw()
        plt.pause(0.03)
    plt.show()
    plt.ioff()
    plt.clf()
