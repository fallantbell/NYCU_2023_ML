import matplotlib.pyplot as plt
import numpy as np

pretrain_train = []
pretrain_test = []
no_pretrain_train = []
no_pretrain_test = []

model = "resnet50"

with open("log/"+model+"_pretrain.txt", 'r', encoding='utf-8') as fp:
    epoch = 0
    for line in fp.readlines():
        s = line.split(' ')
        if s[0] == "train":
            pretrain_train.append(float(s[2]))
        elif s[0] == "test":
            pretrain_test.append(float(s[2]))

with open("log/"+model+".txt", 'r', encoding='utf-8') as fp:
    epoch = 0
    for line in fp.readlines():
        s = line.split(' ')
        if s[0] == "train":
            no_pretrain_train.append(float(s[2]))
        elif s[0] == "test":
            no_pretrain_test.append(float(s[2]))
        

x = np.arange(1,6,1)
        
fig, ax = plt.subplots()

ax.plot(x,pretrain_train,label = "Train(pretrain)")
ax.plot(x,pretrain_test,label = "Test(pretrain)")
ax.plot(x,no_pretrain_train,label = "Train")
ax.plot(x,no_pretrain_test,label = "Test")


ax.legend()
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy(%)")
ax.grid()
ax.set_title(f"{model}")

plt.savefig(f"result/plot_{model}.png")