import torch
import torch.nn as nn
import pickle
import argparse
import math
import numpy as np
from train_CTRdata import LSTMNet
from train_CTRdata import FFNet
from matplotlib import pyplot as plt

if __name__ == '__main__':
    device = "cuda"

    # load trained model
    model = LSTMNet(num_layers=2)
    path = "./checkpoints/CTR_LSTM_L2_bs16_bsfirst/CTR_LSTM_L2_bs16_epoch444_best0.6822978714480996.pt"
    path = "./checkpoints/CTR_LSTM_L2_bs16_bsfirst_ps1/CTR_LSTM_L2_bs16_epoch343_best0.3415350747704506.pt"
    # model = FFNet()
    # path = "./checkpoints/CTR_FFN_L1_HD200_bs16/FFN_bs16_epoch973_ctr_best0.83401919901371.pt"
    # path = "./checkpoints/CTR_FFN_L2_HD64_bs16/FFN_L2_bs16_epoch924_ctr_best0.6523431013231082.pt"
    # path = "./checkpoints/CTR_FFN_L1_HD64_bs16/FFN_L1_bs16_epoch948_ctr_best0.6741083840579938.pt"
    # path = "./checkpoints/CTR_FFN_L1_HD64_bs32/FFN_L1_bs32_epoch997_ctr_best0.6869755158643893.pt"
    # path = "./checkpoints/CTR_FFN_L1_HD64_bs64/FFN_L1_bs64_epoch803_ctr_best0.6977044918707439.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model.cuda()
    model.eval()

    # set test data
    n = 100
    x = np.linspace(-np.pi/2, np.pi/2, n)
    x2 = np.linspace(-np.pi/2, np.pi/2, n)[::-1]
    # y = np.sin(x)*25 - 25
    # y2 = np.sin(x2)*25 - 25
    # print(y[::-1]-y2)
    x_list = list(x)
    x_list.extend(list(x2[1:]))
    y = np.sin(np.array(x_list))

    t = np.linspace(0, 30, n*2-1)
    f = 0.1
    y = np.e**(-0.04*t)*(np.sin(2 * np.pi * f * t - np.pi / 2))
    plt.figure(figsize=(18, 12))
    plt.tick_params(labelsize=30)
    plt.plot(y, 'b', linewidth=3)
    plt.xlabel("Sequence", fontsize=30)
    plt.ylabel("Value", fontsize=30)
    plt.savefig("./results/Descending sinusoidal input.jpg")
    plt.show()
    plt.close()

    joints_list = []
    hidden = (torch.zeros(model.num_layers, 1, model.hidden_dim).to(device),
              torch.zeros(model.num_layers, 1, model.hidden_dim).to(device))
    joints = np.hstack([np.array([y]).T, np.array([np.ones(n*2-1)*(-60)]).T, np.array([np.ones(n*2-1)*0]).T,
                        np.array([np.ones(n*2-1)*(-45)]).T, np.array([np.zeros(n*2-1)]).T, np.array([np.ones(n*2-1)*(-25)]).T]).astype("float32")
    joints_list.append(joints)
    joints = np.hstack([np.array([np.zeros(n * 2 - 1)]).T, np.array([y*60-60]).T, np.array([np.ones(n * 2 - 1) * 0]).T,
                        np.array([np.ones(n * 2 - 1) * (-45)]).T, np.array([np.zeros(n * 2 - 1)]).T,
                        np.array([np.ones(n * 2 - 1) * (-25)]).T]).astype("float32")
    joints_list.append(joints)
    joints = np.hstack([np.array([np.zeros(n * 2 - 1)]).T, np.array([np.ones(n * 2 - 1) * (-60)]).T, np.array([y]).T,
                        np.array([np.ones(n * 2 - 1) * (-45)]).T, np.array([np.zeros(n * 2 - 1)]).T,
                        np.array([np.ones(n * 2 - 1) * (-25)]).T]).astype("float32")
    joints_list.append(joints)
    joints = np.hstack([np.array([np.zeros(n * 2 - 1)]).T, np.array([np.ones(n * 2 - 1) * (-60)]).T, np.array([np.ones(n * 2 - 1) * 0]).T,
                        np.array([y*45-45]).T, np.array([np.zeros(n * 2 - 1)]).T,
                        np.array([np.ones(n * 2 - 1) * (-25)]).T]).astype("float32")
    joints_list.append(joints)
    joints = np.hstack([np.array([np.zeros(n * 2 - 1)]).T, np.array([np.ones(n * 2 - 1) * (-60)]).T, np.array([np.ones(n * 2 - 1) * 0]).T,
                        np.array([np.ones(n * 2 - 1) * (-45)]).T, np.array([y]).T,
                        np.array([np.ones(n * 2 - 1) * (-25)]).T]).astype("float32")
    joints_list.append(joints)
    joints = np.hstack([np.array([np.zeros(n * 2 - 1)]).T, np.array([np.ones(n * 2 - 1) * (-60)]).T, np.array([np.ones(n * 2 - 1) * 0]).T,
                        np.array([np.ones(n * 2 - 1) * (-45)]).T, np.array([np.zeros(n * 2 - 1)]).T,
                        np.array([y*25-25]).T]).astype("float32")
    joints_list.append(joints)

    # plt.figure(figsize=(20, 13))
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    pre_pos = np.array([[0, 0, 0]]).astype("float32")
    # output, h_ = model(torch.tensor([joints]).to(device), hidden)
    for index, label in enumerate(["α1 (rad)", "β1 (mm)", "α2 (rad)", "β2 (mm)", "α3 (rad)", "β3 (mm)"]):
        out = []
        joints = joints_list[index]
        h_ = hidden
        for i in range(joints.shape[0]):
            joint = joints[i:i + 1, :]
            input_ = np.hstack([joint, pre_pos])
            output, h_ = model(torch.tensor([input_]).to(device), h_)
            pre_pos = output.detach().cpu().numpy()[0]
            out.append(pre_pos[0])
        out = np.array(out)
        print(out.shape)

        # plt.subplot(2, 3, index+1)
        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        plt.tick_params(labelsize=18)
        plt.plot(joints[3:-3, index], "b", linewidth=2)
        plt.xlabel("Sequence", fontsize=18)
        plt.ylabel("{}".format(label), fontsize=18)
        plt.title("Input variable: {}".format(label), fontsize=18)
        plt.subplot(1, 2, 2)
        plt.tick_params(labelsize=18)
        # plt.plot(y[3:n], np.linalg.norm(out[:, :]*100, axis=1)[3:n], 'r-', linewidth=4, label="")
        # plt.plot(y[n-1:-3], np.linalg.norm(out[:, :]*100, axis=1)[n-1:-3], 'b-', linewidth=4, label="")
        plt.plot(joints[3:-3, index], np.linalg.norm(out[:, :]*100, axis=1)[3:-3], 'b-', linewidth=2)
        plt.xlabel("{}".format(label), fontsize=18)
        plt.ylabel("Tip position (mm)", fontsize=18)
        plt.title("LSTM Prediction result", fontsize=18)
        # plt.savefig("./results/CTR_data_hysteresis_test{}.jpg".format(index+1))
        plt.close()
    # plt.savefig("./results/CTR_data_hysteresis_test.jpg")
    # plt.show()

    #  validation using ground truth
    csvfile = "C:/Users/wangyuan/Documents/CV/博士申请/Pierre E. Dupont biorobotics/research proposal/concentric tube/CRL-Dataset-CTCR-Pose.csv"
    data = np.loadtxt(csvfile, delimiter=",", skiprows=12500 * 7 + 0, usecols=(0, 1, 2, 3, 4, 5, 33, 34, 35), max_rows=12500)
    joints = data[1:4, :6].astype("float32")
    pre_pos = data[0:1, 6:].astype("float32")

    out = []
    h_ = hidden
    for i in range(joints.shape[0]):
        joint = joints[i:i + 1, :]
        input_ = np.hstack([joint, pre_pos])
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out.append(pre_pos[0])
    out = np.array(out)

    joints_inv = data[1:3, :6].astype("float32")[::-1, :]
    # pre_pos = data[3:4, 6:].astype("float32")
    out2 = []
    h_ = hidden
    for i in range(joints_inv.shape[0]):
        joint = joints_inv[i:i + 1, :]
        input_ = np.hstack([joint, pre_pos])
        output, h_ = model(torch.tensor([input_]).to(device), h_)
        pre_pos = output.detach().cpu().numpy()[0]
        out2.append(pre_pos[0])
    out2 = np.array(out2)

    plt.figure()
    plt.plot(joints[:, 2], np.linalg.norm(out[:, :] * 100, axis=1), 'r-o')
    plt.plot(joints_inv[:, 2], np.linalg.norm(out2[:, :] * 100, axis=1), 'b-o')
    plt.xlabel("α1 (rad)")
    plt.ylabel("Tip position (mm)")
    plt.close()
    # plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    seqr = [0, 12500]
    print(data[seqr[0]:seqr[1], 6:])
    ax.plot(data[seqr[0]:seqr[1], 6], data[seqr[0]:seqr[1], 7], data[seqr[0]:seqr[1], 8], marker="", linewidth=2)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title("Tip position sequence")
    plt.savefig("./results/CTR_data_tip_position_seqlen{}.jpg".format(seqr[1]))
    plt.show()

    plt.figure(figsize=(24, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    seqr = [0, 100]
    data[:, [0,2,4]] = data[:, [0,2,4]]/np.pi*180
    # for index, label in enumerate(["α1 (rad)", "β1 (mm)", "α2 (rad)", "β2 (mm)", "α3 (rad)", "β3 (mm)"]):
    for index, label in enumerate(["α1 (deg)", "β1 (mm)", "α2 (deg)", "β2 (mm)", "α3 (deg)", "β3 (mm)"]):
        plt.subplot(2, 3, index+1)
        plt.tick_params(labelsize=18)
        plt.plot(data[seqr[0]:seqr[1], index], linewidth=2)
        if index % 2 == 0:
            plt.ylim([-65, 65])
        plt.xlabel("Sequence", fontsize=18)
        plt.ylabel("{}".format(label), fontsize=18)
    plt.savefig("./results/CTR_data_sixVariables_seqlen{}.jpg".format(seqr[1]))
    plt.show()
