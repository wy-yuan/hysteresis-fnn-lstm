import numpy as np
import torch
import torch.utils.data as data

class CTRDataset(data.Dataset):

    def __init__(self, stage, seg=50):
        self.stage = stage
        self.seg = seg
        csvfile = "C:/Users/wangyuan/Documents/CV/博士申请/Pierre E. Dupont biorobotics/research proposal/concentric tube/CRL-Dataset-CTCR-Pose.csv"
        # data_all = np.loadtxt(csvfile, delimiter=",", skiprows=0, usecols=(0, 1, 2, 3, 4, 5, 33, 34, 35), max_rows=12500 * 8)

        self.data = []
        if stage == "train":
            self.ctrdata = np.loadtxt(csvfile, delimiter=",", skiprows=0, usecols=(0, 1, 2, 3, 4, 5, 33, 34, 35), max_rows=12500*7)
            for i in range(7):
                self.data.append(self.ctrdata[i*12500:(i+1)*12500, :])
            # index_list = []
            # for i in range(8):
            #     index_list.extend(list(range(i*12500, i*12500+10000)))
            # self.ctrdata = data_all[index_list, :]
        elif stage == "test":
            self.ctrdata = np.loadtxt(csvfile, delimiter=",", skiprows=12500*7+0, usecols=(0, 1, 2, 3, 4, 5, 33, 34, 35), max_rows=12500)
            self.data.append(self.ctrdata)
            # index_list = []
            # for i in range(8):
            #     index_list.extend(list(range(i*12500+10000, i*12500+12500)))
            # self.ctrdata = data_all[index_list, :]


    def __getitem__(self, index):
        rs = 1
        data_ind = np.random.randint(0, len(self.data), 1)[0]
        seq_ind = np.random.randint(1, self.data[data_ind].shape[0] - self.seg * rs, 1)[0]

        joints = np.hstack([self.data[data_ind][[seq_ind + j * rs for j in range(self.seg)], :6],
                            self.data[data_ind][[seq_ind - 1 + j * rs for j in range(self.seg)], 6:] / 100])

        poses = self.data[data_ind][[seq_ind + j * rs for j in range(self.seg)], 6:] / 100

        joints = torch.Tensor(joints).type(torch.float)
        poses = torch.Tensor(poses).type(torch.float)

        return {'joints': joints, 'poses': poses}

    def __len__(self):
        """Return the total number of samples
        """
        return int(len(self.data)*2000)


if __name__ == '__main__':
    csvfile = "C:/Users/wangyuan/Documents/CV/博士申请/Pierre E. Dupont biorobotics/research proposal/concentric tube/CRL-Dataset-CTCR-Pose.csv"
    ctrdata = np.loadtxt(csvfile, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 12, 13, 14), max_rows=12500)
    print(ctrdata.shape)
    print(torch.Tensor(ctrdata[:10, :6]))
    print(torch.Tensor(ctrdata[:10, 6:]))

    # plot histgram
    import matplotlib.pyplot as plt
    plt.hist(ctrdata[:, 5])   # [-1, 1], [-120, 0], [-1, 1], [-90, 0], [-1, 1], [-50, 0]
    plt.show()
