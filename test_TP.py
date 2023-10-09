import torch
import torch.nn as nn
import pickle
import argparse
import math
import numpy as np
from train_TPdata import LSTMNet
from train_TPdata import FFNet
from matplotlib import pyplot as plt
import os

def rmse_norm(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))

if __name__ == '__main__':
    device = "cuda"

    # load trained model
    model = LSTMNet(inp_dim=3, num_layers=2)
    # path = "./checkpoints/TP_LSTM_L2_bs16_freq_Noresample_rs1_MSE_train0baseline/TP_LSTM_L2_bs16_epoch598_best0.0003444615092721119.pt"
    # path = "./checkpoints/TP_LSTM_L2_bs16_train0baseline_lossall_bsfirst_pos0/TP_LSTM_L2_bs16_epoch821_best1.0714787524589833.pt"
    # path = "./checkpoints/TP_LSTM_L2_bs16_train0baseline_lossall_bsfirst_pos1/TP_LSTM_L2_bs16_epoch602_best0.00028749535301764634.pt"
    # path = "./checkpoints/TP_LSTM_L2_bs16_train0baseline_bsfirst_pos0_downsp/TP_LSTM_L2_bs16_epoch378_best0.2629610374569893.pt"
    path = "./checkpoints/TP_LSTM_L2_bs16_train0baseline_bsfirst_pos1_downsp_rs/TP_LSTM_L2_bs16_epoch491_best0.003824349737873203.pt"
    path = "./checkpoints/TP_LSTM_L2_bs16_trainNon0baseline_bsfirst_pos1_downsp_rs/TP_LSTM_L2_bs16_epoch465_best0.003361720456986835.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model.cuda()
    model.eval()

    # set test data
    n = 1
    x = np.linspace(-np.pi/2, np.pi/2, n)
    x2 = np.linspace(-np.pi/2, np.pi/2, n)[::-1]
    y = (np.sin(x)+1)*6
    y2 = (np.sin(x2)+1)*6
    # print(y)
    # print(y2)
    # print(y[::-1]-y2)

    seg = 1
    test_freq = [3]
    path = "./tendon_data/20230928/training_data"
    path = "./tendon_data/20230928/eval_data/"   # 0baseline
    for i, freq in enumerate(test_freq):
        data_path = os.path.join(path, "data_{}.txt".format(str(freq)))
        data = np.loadtxt(data_path, dtype=np.float32, skiprows=1, delimiter=',', unpack=False, encoding='utf-8')
        data = data[::10, :]
        data = data[10:, :]
        data[:, 1] = data[:, 1] - data[0, 1]
        data[:, 6] = data[:, 6] - data[0, 6]
        freqarray = np.ones(data.shape[0]).astype("float32")[:, np.newaxis] * freq / 10
        data = np.hstack([data, freqarray])
        data = np.vstack([np.zeros((seg, 10)), data])
        print(data.shape, max(data[:, 1]), max(data[:, 6]))

    joints = data[:, 1][:, np.newaxis].astype("float32") / 6
    pre_pos = np.array([[0.0]]).astype("float32")
    freq = np.array([[float(test_freq[0]) / 10]]).astype("float32")

    out = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(data.shape[0]):
        joint = joints[i:i + 1, 0:1]
        input_ = np.hstack([joint, pre_pos, freq])
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)

    out2 = []
    joints = data[:, 1][:, np.newaxis].astype("float32") / 6
    pos = data[:, 6][:, np.newaxis].astype("float32") / 100
    freqs = data[:, 9][:, np.newaxis].astype("float32")

    freq = freqs[:seg, 0:1]
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(data.shape[0]-seg):
        joint = joints[i+1:i + seg+1, 0:1]
        pre_pos = pos[i:i+seg, 0:1]
        input_ = np.hstack([joint, pre_pos, freq])
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        predict_pos = output.detach().cpu().numpy()[0]
        out2.append(predict_pos[0, -1])
    out2 = np.array(out2)
    print(out2.shape)

    plt.figure(figsize=(18, 12))
    plt.tick_params(labelsize=30)
    # plt.plot(data[seg:, 1], out2[:] * 100 + 30, 'g--', linewidth=4, label="LSTM prediction")
    plt.plot(data[seg:, 1], out[seg:, 0] * 100 + 30, 'r-', linewidth=4, label="LSTM prediction")
    plt.plot(data[seg:, 1], data[seg:, 6] + 30, 'b-', linewidth=4, label="Ground truth (0.3Hz)")
    plt.xlim([-1, 6])
    plt.ylim([29, 100])
    plt.grid()
    plt.legend(fontsize=30)
    plt.xlabel("Tendon displacement", fontsize=35)
    plt.ylabel("Tip angle X (deg)", fontsize=35)
    plt.title("RMSE:{:.3f} (deg)".format(rmse_norm(out[seg:, 0] * 100, data[seg:, 6])), fontsize=35)
    # plt.savefig("./results/OurTendon-LSTM-0baselineForTrain-Non0Test0.3Hz.jpg")
    plt.savefig("./results/OurTendon-LSTM-Non0baselineForTrain-0Test0.3Hz.jpg")
    plt.show()


