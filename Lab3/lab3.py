import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cpu_num = 2 # Num of CPUs you want to use
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
import torch
torch.set_num_threads(cpu_num)
from dataloader import read_bci_data
from model import EEGNet, DeepConvNet
from record import Record
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torch.optim import Adam

best_acc = {"EEG_Relu":0,"EEG_LeakyRelu":0,"EEG_Elu":0,"Deep_Relu":0,"Deep_LeakyRelu":0,"Deep_Elu":0}

def parse():
    parser = ArgumentParser()
    parser.add_argument('-e',"--epochs",default='300',type = int)
    parser.add_argument('-b',"--batch_size",default='64',type = int)
    parser.add_argument('-lr',"--learning_rate",default='0.001',type = float)
    parser.add_argument('-test',"--test",default='0',type = int)
    return parser

def train_step(model,train_data_loader,loss,optimizer,device,record,activate):
    model.train()

    correct = 0
    for _, (data, target) in enumerate(train_data_loader):
        data = data.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.long)
        output = model(data)

        out_label = output.argmax(dim=1)
        for i in range(len(target)):
            if out_label[i] == target[i]:
                correct += 1

        L = loss(output,target)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()

    train_acc = record.train_accuracy(correct,float(len(train_data_loader.dataset)),activate)
    return train_acc

def test(model,test_data_loader,device,record,activate):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            data = data.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.long)
            output = model(data)

            out_label = output.argmax(dim=1)
            for i in range(len(target)):
                if out_label[i] == target[i]:
                    correct += 1
    test_acc = record.test_accuracy(correct,float(len(test_data_loader.dataset)),activate)
    return test_acc

def main(model,epochs,batch,lr,activate,model_name,record,T=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)

    train_x, train_y, test_x, test_y = read_bci_data()
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    shuffle = True
    train_data_loader = DataLoader(train_data, batch_size=batch, shuffle=shuffle)
    test_data_loader = DataLoader(test_data, batch_size=batch, shuffle=shuffle)

    loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),lr=lr)

    best_test_acc = 0
    
    if T == True:
        test_acc = test(model,test_data_loader,device,record,activate)
        print(f"test_acc = {test_acc}% \n")
        sys.exit(0)
    
    print(f"\n {model_name} train ({activate}): \n")

    for epoch in range(epochs):

        train_acc = train_step(model,train_data_loader,loss,optimizer,device,record,activate)
        
        test_acc = test(model,test_data_loader,device,record,activate)

        print(f"epoch:{epoch} train accuracy = {train_acc}, test accuracy = {test_acc}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_acc[f"{model_name}_{activate}"] = best_test_acc
            if epoch > 100:
                torch.save(model.state_dict(),f"{model_name}_{activate}.pth")




if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    epoch = args.epochs
    batch = args.batch_size
    lr = args.learning_rate
    T = args.test
    activate = ["Relu","LeakyRelu","Elu"]

    if T==1:
        R_EEG = Record("EEG")
        model = EEGNet("Relu")
        model.load_state_dict(torch.load('EEG_Relu.pth'))
        main(model,epoch,batch,lr,"Relu","EEG",R_EEG,True)

    R_EEG = Record("EEG")

    for item in activate:
        model = EEGNet(item)
        main(model,epoch,batch,lr,item,"EEG",R_EEG)

    print("Plot EGG")
    R_EEG.plot_acc(epoch)
        
    R_Deep = Record("Deep")
    for item in activate:
        model = DeepConvNet(item)
        main(model,epoch,batch,lr,item,"Deep",R_Deep)
    
    print("Plot Deep")
    R_Deep.plot_acc(epoch)

    print("\n highest testing accuracy")
    for key in best_acc:
        print(f"{key}: {best_acc[key]}")



