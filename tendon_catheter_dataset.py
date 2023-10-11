import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
import os

class Tendon_catheter_Dataset(data.Dataset):

    def __init__(self, stage, seg=50, filepath="./tendon_data/20230928/training_data", train_freq=[1,2,4,5], test_freq=[3], pos=1):
        self.stage = stage
        self.seg = seg
        self.data = []
        self.pos = pos
        self.train_freq = train_freq  # [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_freq = test_freq  # [0.15, 0.45]
        if stage == "train":
            # path = "./tendon_data/20230928/training_data"
            for i, freq in enumerate(self.train_freq):
                data_path = os.path.join(filepath, "data_{}.txt".format(str(freq)))
                data = self.load_data(data_path, freq)
                self.data.append(data)

        elif stage == "test":
            # path = "./tendon_data/20230928/training_data"
            for i, freq in enumerate(self.test_freq):
                data_path = os.path.join(filepath, "data_{}.txt".format(str(freq)))
                data = self.load_data(data_path, freq)
                self.data.append(data)

    def load_data(self, data_path, freq):
        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        data = data[::10, :]
        data[:, 1] = data[:, 1] - data[0, 1]
        data[:, 6] = data[:, 6] - data[0, 6]
        data = np.hstack([data, np.ones(data.shape[0]).astype("float32")[:, np.newaxis]])  # add flag colum, all set 1
        data[0, 9] = 0  # set initial flag 0
        data = np.vstack([np.zeros((self.seg, 10)), data])
        freqarray = np.ones(data.shape[0]).astype("float32")[:, np.newaxis] * freq / 10  # add frequency colum
        data = np.hstack([data, freqarray])
        print(data.shape, max(data[:, 1]), max(data[:, 6]))
        # self.data.extend([data[j * self.seg:(j + 1) * self.seg] for j in range(data.shape[0] // self.seg - 1)])
        return data

    def __getitem__(self, index):
        # tendon_disp = self.data[index][:, [1]]
        # tip_pos = self.data[index][:, 3:6]
        rs = np.random.randint(1, 10, 1)[0]
        # rs = 1
        if self.stage == "train":
            data_ind = np.random.randint(0, len(self.train_freq), 1)[0]
        elif self.stage == "test":
            data_ind = np.random.randint(0, len(self.test_freq), 1)[0]
        seq_ind = np.random.randint(1, self.data[data_ind].shape[0] - self.seg*rs, 1)[0]
        if self.pos == 1:
            tendon_disp = np.hstack([self.data[data_ind][[seq_ind+j*rs for j in range(self.seg)], 1:2]/6,
                                     self.data[data_ind][[seq_ind-1+j*rs for j in range(self.seg)], 6:7]/100,  # pos
                                     self.data[data_ind][[seq_ind-1+j*rs for j in range(self.seg)], 10:11]])   # freq
        else:
            tendon_disp = np.hstack([self.data[data_ind][[seq_ind + j * rs for j in range(self.seg)], 1:2] / 6,
                                     self.data[data_ind][[seq_ind - 1 + j * rs for j in range(self.seg)], 9:10],   # flag
                                     self.data[data_ind][[seq_ind - 1 + j * rs for j in range(self.seg)], 10:11]])   # freq
        # print(tendon_disp.shape)
        tip_pos = self.data[data_ind][[seq_ind+j*rs for j in range(self.seg)], 6:7]/100

        tendon_disp = torch.Tensor(tendon_disp).type(torch.float)
        tip_pos = torch.Tensor(tip_pos).type(torch.float)

        return {'tendon_disp': tendon_disp, 'tip_pos': tip_pos}

    def __len__(self):
        """Return the total number of samples
        """
        if self.stage == "train":
            data_len = int(len(self.data) * 1000)
        else:
            data_len = int(len(self.data) * 300)
        return data_len

if __name__ == '__main__':
    # data_file = "C:/Users/wangyuan/projects/sequence-model/hysteresis-ffn-lstm/tendon_data/data.txt"
    # data = np.loadtxt(data_file, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    # plt.figure(figsize=(20, 20))
    # plt.plot(data[:, 1], np.linalg.norm(data[:, 3:6], axis=1))
    # plt.show()
    # plt.plot(data[:, 0], np.linalg.norm(data[:, 3:6], axis=1))
    # plt.show()
    '''
    path = "C:/Users/wangyuan/projects/sequence-model/hysteresis-ffn-lstm/tendon_data/all"

    data_path = os.path.join(path, "data_ND_Mean37.5_0.02.txt")
    data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    plt.figure(figsize=(20, 12))
    plt.plot(data[:, 1]-data[0,1], np.linalg.norm(data[:, 3:6], axis=1))
    plt.title("data_ND_Mean37.5_0.02")
    plt.savefig('./results/tendondata_ND_Mean37.5_0.02.jpg')
    # plt.show()

    Decay_tag, Converging_deg, freq = ["ND", "D"], [0, 45], [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5]
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(len(freq)):
        data_path = os.path.join(path, "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i])))
        data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        plt.subplot(3, 3, i+1)
        plt.plot(data[:, 1]-data[0,1], np.linalg.norm(data[:, 3:6], axis=1))
        plt.xlabel("Tendon displacement")
        plt.ylabel("Tip position")
        plt.xlim([-1, 13])
        plt.ylim([-1, 75])
        plt.grid()
        plt.title("{} Converging_deg:{} frequency:{}".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i])))
    plt.savefig('./results/tendondata_D_Conv0.jpg')
    # plt.show()
    #
    # # plot the time distance histogram
    # plt.figure(figsize=(20, 12))
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # for i in range(len(freq)):
    #     data_path = os.path.join(path,
    #                              "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i])))
    #     data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    #     plt.subplot(3, 3, i + 1)
    #     # plt.plot(data[:, 1] - data[0, 1], np.linalg.norm(data[:, 3:6], axis=1))
    #     plt.hist(data[1:, 0]-data[:-1, 0])
    #     plt.title("{} Conv_deg:{} freq:{} len:{}".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i]), str(len(data[:, 0]))))
    # plt.savefig('./results/tendondata_D_Conv0_timehist.jpg')
    # plt.show()

    # plot the time-tip position
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(len(freq)):
        data_path = os.path.join(path,
                                 "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i])))
        data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        plt.subplot(3, 3, i + 1)
        plt.plot(data[:, 0]*(100000), np.linalg.norm(data[:, 3:6], axis=1))
        plt.xlabel("Time")
        plt.ylabel("Tip position")
        plt.title("{} Conv_deg:{} freq:{}".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i])))
    plt.savefig('./results/tendondata_D_Conv0_curve.jpg')
    # plt.show()

    # plot the motor current - time
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(len(freq)):
        data_path = os.path.join(path,
                                 "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i])))
        data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        plt.subplot(3, 3, i + 1)
        plt.plot(data[:, 0]*(100000), data[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Motor current")
        plt.title("{} Conv_deg:{} freq:{}".format(Decay_tag[1], str(Converging_deg[0]), str(freq[i])))
    plt.savefig('./results/tendondata_D_Conv0_motorcurrent.jpg')
    # plt.show()

###############################################################################################################
    Decay_tag, Converging_deg, freq = ["ND", "D"], [0, 45], [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5]
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(len(freq)):
        data_path = os.path.join(path,
                                 "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i])))
        data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        plt.subplot(3, 3, i + 1)
        plt.plot(data[:, 1]-data[0,1], np.linalg.norm(data[:, 3:6], axis=1))
        plt.xlabel("Tendon displacement")
        plt.ylabel("Tip position")
        plt.xlim([-1, 13])
        plt.ylim([-1, 75])
        plt.grid()
        plt.title("{} Converging_deg:{} frequency:{}".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i])))
    plt.savefig('./results/tendondata_D_Conv45.jpg')
    # plt.show()
    #
    # plt.figure(figsize=(20, 12))
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # for i in range(len(freq)):
    #     data_path = os.path.join(path,
    #                              "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i])))
    #     data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
    #     plt.subplot(3, 3, i + 1)
    #     # plt.plot(data[:, 1] - data[0, 1], np.linalg.norm(data[:, 3:6], axis=1))
    #     plt.hist(data[1:, 0]-data[:-1, 0])
    #     plt.title("{} Conv_deg:{} freq:{} len:{}".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i]), str(len(data[:, 0]))))
    # plt.savefig('./results/tendondata_D_Conv45_timehist.jpg')
    # plt.show()

    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(len(freq)):
        data_path = os.path.join(path,
                                 "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i])))
        data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        plt.subplot(3, 3, i + 1)
        plt.plot(data[:, 0]*(100000), np.linalg.norm(data[:, 3:6], axis=1))
        plt.xlabel("Time")
        plt.ylabel("Tip position")
        plt.title("{} Conv_deg:{} freq:{}".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i])))
    plt.savefig('./results/tendondata_D_Conv45_curve.jpg')
    # plt.show()

    # plot the motor current - time
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(len(freq)):
        data_path = os.path.join(path,
                                 "data_{}_Conv{}_{}.txt".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i])))
        data = np.loadtxt(data_path, dtype=float, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        plt.subplot(3, 3, i + 1)
        plt.plot(data[:, 0] * (100000), data[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Motor current")
        plt.title("{} Conv_deg:{} freq:{}".format(Decay_tag[1], str(Converging_deg[1]), str(freq[i])))
    plt.savefig('./results/tendondata_D_Conv45_motorcurrent.jpg')
    # plt.show()
    '''

    plt.figure(figsize=(18, 12))
    plt.tick_params(labelsize=30)
    test_freq = [1, 2, 3, 4, 5]
    path = "./tendon_data/20230928/training_data"
    for i, freq in enumerate(test_freq):
        data_path = os.path.join(path, "data_{}.txt".format(str(freq)))
        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        # data[:, 1] = data[:, 1] - data[0, 1]
        # data[:, 6] = data[:, 6] - data[0, 6]
        plt.plot(data[:, 1], data[:, 6], linewidth=2, label="Non-zero baseline {}Hz".format(freq/10))

    test_freq = [5]
    path = "./tendon_data/20230928/eval_data/"
    for i, freq in enumerate(test_freq):
        data_path = os.path.join(path, "data_{}.txt".format(str(freq)))
        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        # data[:, 1] = data[:, 1] - data[0, 1]
        # data[:, 6] = data[:, 6] - data[0, 6]
        plt.plot(data[:, 1], data[:, 6], '--', linewidth=3, label="Zero baseline {}Hz".format(freq / 10))
    plt.xlim([-1, 6])
    plt.ylim([29, 100])
    plt.grid()
    plt.xlabel("Tendon displacement", fontsize=30)
    plt.ylabel("Tip angle X (deg)", fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig("./results/tendon_tipAng-tendondisp_zerobaseline{}Hz.jpg".format(test_freq[0]/10))
    plt.show()




