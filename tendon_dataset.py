import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d


class tendonDataset(data.Dataset):

    def __init__(self, stage, seg=50):
        self.stage = stage
        self.seg = seg
        self.data = []
        path = "C:/Users/wangyuan/Documents/CV/BCH/data from Yash/Frequency_Trials/"
        self.train_freq = ["10", "20", "30"]  # [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_freq = ["15"]  # [0.15, 0.45]
        if stage == "train":
            for i, freq in enumerate(self.train_freq):
                matfile = os.path.join(path, "f0{}_data.mat".format(freq))
                # self.data.extend([data[j * self.seg:(j + 1) * self.seg] for j in range(data.shape[0] // self.seg - 1)])
                self.data.append(self.load_data(matfile, float(freq)/100))

        elif stage == "test":
            for i, freq in enumerate(self.test_freq):
                matfile = os.path.join(path, "f0{}_data.mat".format(freq))
                self.data.append(self.load_data(matfile, float(freq)/100))

    def load_data(self, filename, freq, resample=True, resample_num=2000):
        data = loadmat(filename)
        data_tension = data["Tension"][:, 0]
        # data_tipdist = data["tip_marker_distance"][0]
        data_tipdist = np.linalg.norm(data["tip_marker"][:, :] - data["tip_marker"][0, :], axis=1)
        if resample:
            xp = np.linspace(0, len(data_tension) - 1, resample_num)
            tension_lin = interp1d(np.arange(len(data_tension)), data_tension)
            tension_resample = tension_lin(xp)
            tipdist_lin = interp1d(np.arange(len(data_tipdist)), data_tipdist)
            tipdist_resample = tipdist_lin(xp)
        else:
            tension_resample = data_tension
            tipdist_resample = data_tipdist

        tension = np.vstack([np.zeros(self.seg)[:, np.newaxis], tension_resample[:, np.newaxis]])
        tip_marker_distance = np.vstack([np.zeros(self.seg)[:, np.newaxis], tipdist_resample[:, np.newaxis]])
        freqarray = np.ones_like(tension)*freq
        return np.hstack([tension, tip_marker_distance, freqarray])

    def __getitem__(self, index):
        # tendon_disp = self.data[index][:, [1]]
        # tip_pos = self.data[index][:, 3:6]
        # rs = np.random.randint(1, 10, 1)[0]  # set random resample rate
        rs = 1
        if self.stage == "train":
            data_ind = np.random.randint(0, len(self.train_freq), 1)[0]
        elif self.stage == "test":
            data_ind = np.random.randint(0, len(self.test_freq), 1)[0]
        seq_ind = np.random.randint(1, self.data[data_ind].shape[0] - self.seg * rs, 1)[0]

        tendon_disp = np.hstack([self.data[data_ind][[seq_ind + j * rs for j in range(self.seg)], 0:1] / 5,
                                 self.data[data_ind][[seq_ind - 1 + j * rs for j in range(self.seg)], 1:2] / 100,
                                 self.data[data_ind][[seq_ind - 1 + j * rs for j in range(self.seg)], 2:3]])
        # print(tendon_disp.shape)
        tip_pos = self.data[data_ind][[seq_ind + j * rs for j in range(self.seg)], 1:2] / 100

        tendon_disp = torch.Tensor(tendon_disp).type(torch.float)
        tip_pos = torch.Tensor(tip_pos).type(torch.float)

        return {'tendon_disp': tendon_disp, 'tip_pos': tip_pos}

    def __len__(self):
        """Return the total number of samples
        """
        return int(len(self.data) * 1000)

def test_data1():
    # matfile = "C:/Users/wangyuan/Documents/CV/BCH/data from Yash/0-90 deg tests/0-90 deg tests/Test_1_data/Test_1_data.mat"
    matfile = "C:/Users/wangyuan/Documents/CV/BCH/data from Yash/OneDrive_1_2023-9-26/Data_x2_09252023.mat"
    plt.figure(figsize=(18, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tick_params(labelsize=30)
    for i, freq in enumerate(["10", "15", "20", "30"]):
        path = "C:/Users/wangyuan/Documents/CV/BCH/data from Yash/Frequency_Trials/"
        matfile = os.path.join(path, "f0{}_data.mat".format(freq))
        data = loadmat(matfile)
        print(data.keys())
        tension = data["Tension"]
        time = data["Time"]
        tip_marker = data["tip_marker"]
        tip_marker_distance = data["tip_marker_distance"]
        tip_marker_ref = data["tip_marker_ref"]
        Stroke = data["Stroke"]
        Stroke_ref = data["Stroke_Ref"]
        Last_Marker = data["MarkerDataSynced"][:, :, -1]
        print(tension.shape, time.shape, tip_marker.shape, tip_marker_distance.shape, tip_marker_ref.shape, Stroke.shape, Stroke_ref.shape)

        # print(tip_marker_ref)
        # print("________", Stroke_ref)
        time_interval = time[1:, 0] - time[:-1, 0]
        # print(time_interval)

        xp = np.linspace(0, len(tension[:,0]) - 1, 100)
        tension_lin = interp1d(np.arange(len(tension[:, 0])), tension[:, 0])
        tension_resample = tension_lin(xp)
        # plt.plot(tension_resample, '-', linewidth=4, label="decaying 0 - 0.{} Hz".format(freq))
        # plt.plot(time[:, 0], tip_marker[:, 0], '-', label='tip_marker X')
        # plt.plot(time[:, 0], tip_marker[:, 1], '-', label='tip_marker Y')
        # plt.plot(time[:, 0], tip_marker[:, 2], '-', label='tip_marker Z')
        # plt.plot(time[:, 0], tip_marker_distance[0], '-', label='tip_marker dist')
        # # plt.plot(time[:, 0], tip_marker_ref[0, 0], '-', label='tip_marker ref')
        # # plt.ylim([0, 10])
        # plt.xlabel('', fontsize=35)
        # plt.ylabel('Tension (N)', fontsize=35)
        # plt.legend(fontsize=30)
        # # plt.savefig('./results/tendondata.jpg')
        # plt.show()

        # plt.subplot(2, 2, i + 1)
        # plt.plot(tension[:, 0], np.linalg.norm(tip_marker[:, :]-tip_marker[0, :], axis=1), linewidth=4, label="decaying 0 - 0.{} Hz".format(freq))
        # plt.plot(tension[:, 0], tip_marker_distance[0], '-', linewidth=4, label="0.{} Hz".format(freq))
        # plt.xlabel('Tendon Tension (N)', fontsize=35)
        # plt.ylabel('Tip displacement (mm)', fontsize=35)
        # plt.xlim([-1, 7])
        # plt.ylim([-10, 100])
        # plt.legend(fontsize=30)

        # plt.title("frequency:0.{}".format(freq))

        # plt.plot(Stroke[:, 0], np.linalg.norm(tip_marker[:, :] - tip_marker[0, :], axis=1))
        # plt.plot(Stroke[:, 0], tip_marker_distance[0])
        # plt.plot(Stroke[:, 0], tip_marker_distance[0], '-', linewidth=4, label="0.{} Hz".format(freq))
        # plt.xlabel('Tendon displacement', fontsize=35)
        # plt.ylabel('Tip displacement (mm)', fontsize=35)
        # plt.xlim([-1, 10])
        # plt.ylim([-10, 100])
        # plt.legend(fontsize=30)
        # plt.grid()
        # plt.title("frequency:0.{}".format(freq))

    for i, freq in enumerate(["02"]):
        path = "C:/Users/wangyuan/Documents/CV/BCH/data from Yash/Frequency_Trials/"
        matfile = os.path.join(path, "f0{}_data_2.mat".format(freq))
        data = loadmat(matfile)
        tension = data["Tension"]
        time = data["Time"]
        tip_marker = data["tip_marker"]
        xp = np.linspace(0, len(tension[:-50, 0]) - 1, 100)
        tension_lin = interp1d(np.arange(len(tension[:-50, 0])), tension[:-50, 0])
        tension_resample = tension_lin(xp)
        # plt.plot(tension[:, 0], np.linalg.norm(tip_marker[:, :]-tip_marker[0, :], axis=1), linewidth=4, label="decaying 45 - 0.{} Hz".format(freq))
        # plt.plot(tension_resample, '-', linewidth=4, label="decaying 0 - 0.{} Hz".format(freq))
        # plt.plot(time, tension, '-', linewidth=2, label="decaying 45 - 0.{} Hz".format(freq))


    path = "C:/Users/wangyuan/Documents/CV/BCH/data from Yash/Frequency_Trials/"
    matfile = os.path.join(path, "Random_data.mat".format(freq))
    data = loadmat(matfile)
    tension = data["Tension"]
    time = data["Time"]
    tip_marker = data["MarkerDataSynced"][:, :, -1]
    out_0 = np.load("./results/Tend_0baselineForTrain_random_out.npy")
    out_Non0 = np.load("./results/Tend_Non0baselineForTrain_random_out.npy")
    # plt.plot(tension[:, 0], np.linalg.norm(tip_marker[:, :] - tip_marker[0, :], axis=1), linewidth=4, label="Random data")
    plt.plot(time, np.linalg.norm(tip_marker[:, :] - tip_marker[0, :], axis=1), linewidth=3, label="Ground Truth (Random)")
    # plt.plot(time, out_0, linewidth=3, label="LSTM model")
    plt.plot(time, out_Non0, linewidth=3, label="LSTM model")
    plt.xlabel('Time (s)', fontsize=35)
    plt.ylabel('Tip displacement (mm)', fontsize=35)
    plt.ylim([-2, 102])
    plt.legend(fontsize=20)
    # plt.savefig('./results/Tip_displacement_vs_time_random_input.jpg')
    plt.savefig('./results/Tip_displacement_vs_time_random&LSTM2.jpg')
    plt.show()
    # plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    # plt.legend(fontsize=25)
    ax.plot(tip_marker[:, 0], tip_marker[:, 1], tip_marker[:, 2], marker="", linewidth=2)
    ax.set_xlabel('X (mm)', fontsize=15)
    ax.set_ylabel('Y (mm)', fontsize=15)
    ax.set_zlabel('Z (mm)', fontsize=15)
    ax.set_title("Tip position sequence", fontsize=20)
    plt.grid()
    # plt.savefig('./results/Tip_position_random_input.jpg')
    # plt.savefig('./results/tendon_tension-tip_disp_allfreq.jpg')
    # plt.savefig('./results/tendon_disp-tip_disp_0.02Hz.jpg')
    # plt.savefig('./results/Tendon_tension-time_0.02.jpg')
    # plt.savefig('./results/tendondata_stroke-tippos.jpg')
    plt.show()
    # plt.figure(figsize=(20, 20))
    # plt.plot(Stroke[:, 0]-Stroke_ref[:, 0], np.linalg.norm(tip_marker[:, :] - tip_marker[0, :], axis=1))
    # plt.xlabel('Stroke-ref')
    # plt.ylabel('Tip displacement (mm)')
    # plt.show()


if __name__ == '__main__':
    test_data1()

