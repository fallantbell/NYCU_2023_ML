import matplotlib.pyplot as plt
import numpy as np

KL_loss = []
psnr = []
x = []
x_psnr = []

model = "Cyclical_tfr"

with open("logs/fp/"+model+"/train_record.txt", 'r', encoding='utf-8') as fp:
    epoch = 0
    for line in fp.readlines():
        s = line.split(' ')
        if s[0] == "[epoch:":
            x.append(epoch)
            KL_loss.append(float(s[11]))
            epoch+=1
        if s[1] == "validate":
            x_psnr.append(epoch-1)
            psnr.append(float(s[4]))
        
fig, ax = plt.subplots()

# plot KL_loss
ax.plot(x, KL_loss, label="KL_loss")

# create a secondary axis for psnr
ax_psnr = ax.twinx()

# plot psnr
ax_psnr.plot(x_psnr, psnr,marker="o", label="psnr", color='orange')

# set labels and title
ax.legend(loc="upper left")
ax_psnr.legend(loc="upper right")
ax.set_xlabel("Epochs")
ax.set_ylabel("KL_loss")
ax_psnr.set_ylabel("psnr")
ax.grid()
ax.set_title(f"{model}")

plt.savefig(f"plot_result/plot_{model}.png")