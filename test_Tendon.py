import torch
import torch.nn as nn
import pickle
import argparse
import math
import numpy as np
from train_Tendon import LSTMNet
from train_Tendon import FFNet
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d

def load_data(filename, seg, resample=True):
    data = loadmat(filename)
    print(data.keys())
    data_time = data["Time"]
    data_tension = data["Tension"][:, 0]
    Last_Marker = data["MarkerDataSynced"][:,:,-1]

    # data_tipdist = data["tip_marker_distance"][0]
    # data_tipdist = np.linalg.norm(data["tip_marker"][:,:] - data["tip_marker"][0, :], axis=1)
    data_tipdist = np.linalg.norm(Last_Marker[:,:] - Last_Marker[0, :], axis=1)
    print(data_tension.shape, data_time.shape, Last_Marker.shape)

    if resample:
        xp = np.linspace(0, len(data_tension) - 1, 200)
        tension_lin = interp1d(np.arange(len(data_tension)), data_tension)
        tension_resample = tension_lin(xp)
        tipdist_lin = interp1d(np.arange(len(data_tipdist)), data_tipdist)
        tipdist_resample = tipdist_lin(xp)
    else:
        tension_resample = data_tension
        tipdist_resample = data_tipdist

    tension = np.vstack([np.zeros(seg)[:, np.newaxis], tension_resample[:, np.newaxis]])
    tip_marker_distance = np.vstack([np.zeros(seg)[:, np.newaxis], tipdist_resample[:, np.newaxis]])
    return tension, tip_marker_distance, data_time

def rmse_norm(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))

if __name__ == '__main__':
    device = "cuda"
    # load trained model
    model = LSTMNet(inp_dim=3, num_layers=2)
    # path = "./checkpoints/Tendon_LSTM_L2_bs16_train01-15-20/Tend_LSTM_L2_bs16_epoch450.pt"
    # path = "./checkpoints/Tendon_LSTM_L2_bs16_train01-20-30/Tend_LSTM_L2_bs16_epoch500.pt"
    # path = "./checkpoints/Tend_LSTM_L2_bs16_train10-20-30_freq_resample/Tend_LSTM_L2_bs16_epoch500.pt"
    # path = "./checkpoints/Tend_LSTM_L2_bs16_train10-20-30_freq_RDresample/Tend_LSTM_L2_bs16_epoch500.pt"
    # path = "./checkpoints/Tend_LSTM_L2_bs16_train10-20-30_freq_Noresample/Tend_LSTM_L2_bs16_epoch470_best0.0035686560736466495.pt"
    path = "./checkpoints/Tend_LSTM_L2_bs16_train10-20-30_bsfirst/Tend_LSTM_L2_bs16_epoch500.pt"
    path = "./checkpoints/Tend_LSTM_L2_bs16_train10-20-30_bsfirst_lossall/Tend_LSTM_L2_bs16_epoch191_best0.18772485107183456.pt"
    path = "./checkpoints/Tend_LSTM_L2_bs16_train10-20-30_bsfirst_lossall_pos0/Tend_LSTM_L2_bs16_epoch50.pt"
    path = "./checkpoints/Tend_LSTM_L2_bs16_traindeg45_bsfirst_lossall_pos0/Tend_LSTM_L2_bs16_epoch300.pt"
    path = "./checkpoints/Tend_LSTM_L2_bs16_traindeg45_bsfirst_pos0_rs/Tend_LSTM_L2_bs16_epoch100.pt"
    path = "./checkpoints/Tend_LSTM_L2_bs16_traindeg0_bsfirst_pos0_rs/Tend_LSTM_L2_bs16_epoch74_best2.651146920900496.pt"
    # model = FFNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.cuda()
    model.eval()

    path = "C:/Users/wangyuan/Documents/CV/BCH/data from Yash/Frequency_Trials/"
    seg = 1
    test_freq = ["15"]
    matfile = os.path.join(path, "f0{}_data_2.mat".format(test_freq[0]))
    matfile = os.path.join(path, "Random_data.mat")
    tension, tip_marker_distance, time = load_data(matfile, seg, resample=True)

    joints = tension.astype("float32") / 5
    pre_pos = np.array([[0.0]]).astype("float32")
    freq = np.array([[float(test_freq[0])/100]]).astype("float32")

    out, out2 = [], []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    h_ = hidden
    for i in range(tension.shape[0]):
        joint = joints[i:i+1, 0:1]
        input_ = np.hstack([joint, pre_pos, freq])
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)
    np.save("./results/Tend_Non0baselineForTrain_random_out.npy", out[seg:, 0]*100)

    out2 = []
    joints = tension.astype("float32") / 5
    pos = tip_marker_distance.astype("float32") / 100
    freqs = np.ones_like(pos).astype("float32")*float(test_freq[0])/100
    freq = freqs[:seg, 0:1]
    pre_pos = pos[:seg, 0:1]
    h_ = hidden
    for i in range(tension.shape[0] - seg):
        joint = joints[i + 1:i + seg + 1, 0:1]
        # pre_pos = pos[i:i+seg, 0:1]
        input_ = np.hstack([joint, pre_pos*0, freq])
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        predict_pos = output.detach().cpu().numpy()[0]
        pre_pos = predict_pos
        out2.append(predict_pos[0, -1])
    out2 = np.array(out2)

    print(out2.shape)

    plt.figure(figsize=(18, 12))
    plt.tick_params(labelsize=30)
    plt.plot(tension[seg:, 0], out[seg:, 0]*100, 'r-', linewidth=4, label="LSTM prediction")
    # plt.plot(tension[seg:, 0], out2[:]*100, 'g--', linewidth=4, label="LSTM prediction")
    plt.plot(tension[seg:, 0], tip_marker_distance[seg:, 0], 'b-', linewidth=4, label="Ground truth (Random)")
    plt.xlim([0, 5])
    plt.ylim([-1, 100])
    plt.grid()
    plt.legend(fontsize=30)
    plt.xlabel("Tendon tension (N)", fontsize=35)
    plt.ylabel("Tip displacement (mm)", fontsize=35)
    plt.title("RMSE:{:.3f} (mm)".format(rmse_norm(out[seg:, 0]*100, tip_marker_distance[seg:, 0])), fontsize=35)
    # plt.savefig("./results/Tendon-LSTM-0baselineForTrain-Non0Test0.02Hz.jpg")
    # plt.savefig("./results/Tendon-LSTM-Non0baselineForTrain-0Test0.15Hz.jpg")
    # plt.savefig("./results/Tendon-LSTM-Non0baselineForTrain-TestRandom.jpg")
    # plt.savefig("./results/Tendon-LSTM-0baselineForTrain-TestRandom.jpg")
    plt.show()

    # plt.figure(figsize=(18, 12))
    # plt.tick_params(labelsize=30)
    # plt.plot(time, tension[seg:,:], "-o")
    # plt.grid()
    # plt.show()


