import pickle
from matplotlib import pyplot as plt
import numpy as np

name = "FFN_acc_bs16_epoch1000_ctr"
# name = "FFN_acc_bs64_epoch1000_ctr"
# name = "LSTM_L1_acc_bs32_epoch1000_ctr"
# name = "LSTM_L1_acc_bs16_epoch1000_ctr"
# acc_file = "./results/acc_{}.pkl".format(name)
acc_file = "./checkpoints/{}.pkl".format(name)
acc = pickle.load(open(acc_file, "rb"))
# acc = pickle.load( open( "./acc.pkl".format(name), "rb" ) )
# acc1 = pickle.load( open( "acc1.pkl", "rb" ) )


plt.figure(figsize=(20, 10))
plt.plot(acc['train'], '-', label='train')
plt.plot(acc['test'], '-', label='test')
plt.ylim([0, 20])
plt.xlabel('Epochs')
plt.ylabel('Error (mm)')
plt.grid()
plt.legend()
plt.savefig('./results/acc_{}.jpg'.format(name))
plt.show()