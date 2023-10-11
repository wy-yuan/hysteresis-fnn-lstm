import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import argparse
import math
import numpy as np
from torch.utils.data import DataLoader
from tendon_polymer_dataset import Tendon_polymer_Dataset
from tendon_catheter_dataset import Tendon_catheter_Dataset
import os

criterionMSE = nn.MSELoss()

class LSTMNet(nn.Module):

    def __init__(self, inp_dim=3, hidden_dim=64, num_layers=3, dropout=0, act=None):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 1
        self.act = act

        self.rnn = nn.LSTM(inp_dim, self.hidden_dim, dropout=dropout, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        # nn.init.orthogonal(self.rnn.weight_ih_l0)
        # nn.init.orthogonal(self.rnn.weight_hh_l0)

    def forward(self, points, hidden):
        lstm_out, h_ = self.rnn(points, hidden)
        if self.act == "tanh":
            lstm_out = F.tanh(lstm_out)
        elif self.act == "relu":
            lstm_out = F.relu(lstm_out)
        linear_out = self.linear(lstm_out)
        # out = F.log_softmax(linear_out, dim=1)
        return linear_out, h_

class FFNet(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=64, dropout=0):
        super(FFNet, self).__init__()
        # self.num_layers = 5
        self.output_dim = 3
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


def train(args, model, device, train_loader, optimizer, epoch, clip_value=100):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        tendon_disp, tip_pos = data["tendon_disp"].to(device), data["tip_pos"].to(device)
        optimizer.zero_grad()
        # print("**********", tendon_disp.shape)
        bs = tendon_disp.shape[0]
        hidden = (torch.zeros(model.num_layers, bs, model.hidden_dim).to(device),
                  torch.zeros(model.num_layers, bs, model.hidden_dim).to(device))
        output, h_ = model(tendon_disp, hidden)

        output = torch.transpose(output, 1, 2)
        poses = torch.transpose(tip_pos, 1, 2)
        # print("*******", poses.shape)
        # loss = F.pairwise_distance(output[:, :, :], poses[:, :, :], p=2)
        # loss = torch.mean(loss)
        loss = criterionMSE(output, poses)*10000
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Batch: {} \tLoss: {:.6f}'.format(
                batch_idx, loss.item()))
    return loss.item()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for b_idx, data in enumerate(test_loader):
            tendon_disp, tip_pos = data["tendon_disp"].to(device), data["tip_pos"].to(device)
            bs = tendon_disp.shape[0]
            hidden = (torch.zeros(model.num_layers, bs, model.hidden_dim).to(device),
                      torch.zeros(model.num_layers, bs, model.hidden_dim).to(device))
            output, h_ = model(tendon_disp, hidden)
            # print("test **********", output.size())
            output = torch.transpose(output, 1, 2)
            poses = torch.transpose(tip_pos, 1, 2)
            # print(poses.shape)
            # loss = F.pairwise_distance(output[:,:,:], poses[:,:,:], p=2)
            # loss = torch.mean(loss)
            loss = criterionMSE(output, poses)*10000
            test_loss += loss.item()
            total += 1
    test_loss /= total

    print('\n--Test set: Average loss: {:.4f}\n'.format(
        test_loss))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_name', type=str, default="LSTM")
    parser.add_argument('--checkpoints_dir', type=str,
                        default="./checkpoints/TP_LSTM_L2_bs16_trainAll_bsfirst_pos0_downsp_rs_tanh/")
    parser.add_argument('--lstm_layers', type=int, default=2)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    filepath = "./tendon_data/20230928/all_data"
    train_f = [1, 2, 4, 5, 6, 7, 9, 10]
    test_f = [3, 8]
    pos = 0
    act = "tanh"
    lstm_test_acc = []
    lstm_train_loss = []
    if "LSTM" in args.model_name:
        print('Training LSTM.')
        model = LSTMNet(inp_dim=3, num_layers=args.lstm_layers, act=act).to(device)
        lr = 10 * 1e-4  # 10 * 1e-4
    else:
        print('Training FFN.')
        model = FFNet().to(device)
        lr = 5 * 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # lr 5*1e-4 for FFN, 10*1e-4 for LSTM
    if "LSTM" in args.model_name:
        train_dataset = Tendon_catheter_Dataset("train", filepath=filepath, train_freq=train_f, test_freq=test_f, pos=pos)
        test_dataset = Tendon_catheter_Dataset("test", filepath=filepath, train_freq=train_f, test_freq=test_f, pos=pos)
    else:
        train_dataset = Tendon_catheter_Dataset("train", seg=1, filepath=filepath, train_freq=train_f, test_freq=test_f, pos=pos)
        test_dataset = Tendon_catheter_Dataset("test", seg=1, filepath=filepath, train_freq=train_f, test_freq=test_f, pos=pos)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    min_test_acc = 1000
    for epoch in range(1, args.epochs + 1):
        print('------Train epoch---------: {} \n'.format(epoch))
        train_acc = train(args, model, device, train_data, optimizer, epoch)
        test_acc = test(model, device, test_data)

        lstm_test_acc.append(test_acc)
        lstm_train_loss.append(train_acc)
        if args.save_model:
            if epoch % 50 == 0:
                torch.save(model.state_dict(), args.checkpoints_dir +
                           "TP_" + args.model_name + "_L{}_bs{}_epoch{}.pt".format(str(args.lstm_layers),
                                                                                     str(args.batch_size), str(epoch)))
            if epoch > 10 and test_acc < min_test_acc:
                torch.save(model.state_dict(), args.checkpoints_dir +
                           "TP_" + args.model_name + "_L{}_bs{}_epoch{}_best{}.pt".format(str(args.lstm_layers),
                                                                                            str(args.batch_size),
                                                                                            str(epoch), str(test_acc)))
                min_test_acc = test_acc

    print(model)
    pickle.dump( {'test': lstm_test_acc, 'train': lstm_train_loss},
                 open(args.checkpoints_dir + "TP_" + args.model_name + "_L{}_acc_bs{}_epoch{}.pkl".format(
                     str(args.lstm_layers), str(args.batch_size), str(args.epochs)), "wb"))

if __name__ == '__main__':
    main()

    # output = torch.tensor(np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,1,0],[0,0,2],[3,0,0],[4,0,0]]])).type(torch.float)
    # poses = torch.tensor(np.array([[[0,1,0],[0,0,2],[3,0,0],[4,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]])).type(torch.float)
    # print(output.shape)
    # print(poses.shape)
    # output = torch.transpose(output, 1, 2)
    # poses = torch.transpose(poses, 1, 2)
    # loss = F.pairwise_distance(output, poses, p=2)
    # loss = torch.mean(loss)
    # print(loss)