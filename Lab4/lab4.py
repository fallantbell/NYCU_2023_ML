import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from dataloader import RetinopathyLoader
from torch.utils.data import Dataset, DataLoader
from model import ResNet18,ResNet50,ResNet18_pretrain,ResNet50_pretrain
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
import argparse
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--eva', type=int, default=0)

    args = parser.parse_args()

    return args

def test(model,test_data_loader,device,epoch,eva=0):
    model.eval()
    correct = 0
    GT_list = []
    predict_list = []
    with torch.no_grad():
        for i,(image,label) in enumerate(tqdm(test_data_loader,desc = f"Testing epoch:{epoch+1}")):
            image = image.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            output = model(image)

            out_label = output.argmax(dim=1)
            for i in range(len(label)):
                if out_label[i] == label[i]:
                    correct += 1
                if eva == 1:
                    GT_list.append(label[i].to('cpu').numpy())
                    predict_list.append(out_label[i].to('cpu').numpy())

    accuracy = 100. * correct/float(len(test_data_loader.dataset))
    print(f"test accuracy: {accuracy}")

    return accuracy,GT_list,predict_list
            

def training(batch,epochs,lr,model,device , model_name):
    
    training_data = RetinopathyLoader("new_train_resize512","train")
    train_dataloader = DataLoader(training_data,batch_size=batch,num_workers=4,shuffle=True,pin_memory=True)
    
    test_data = RetinopathyLoader("new_test_resize512","test")
    test_dataloader = DataLoader(test_data,batch_size=batch,num_workers=4,shuffle=True,pin_memory=True)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)

    best_acc = 0.0

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        correct = 0
        print(f"Epoch {epoch+1}")
        for i,(image,label) in enumerate(tqdm(train_dataloader,desc = f"Training epoch:{epoch+1}")):
            image = image.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            output = model(image)

            out_label = output.argmax(dim=1)
            for i in range(len(label)):
                if out_label[i] == label[i]:
                    correct += 1
            
            L = loss(output,label)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
        accuracy = 100. * correct/float(len(train_dataloader.dataset))
        print(f"train accuracy: {accuracy}")
        
        test_acc, _ , _ = test(model,test_dataloader,device,epoch)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),f"{model_name}_weight.pth")

def evaluate(model,batch,device,model_name):
    test_data = RetinopathyLoader("new_test_resize512","test")
    test_dataloader = DataLoader(test_data,batch_size=batch,num_workers=4,shuffle=True,pin_memory=True)
    model = model.to(device)

    test_acc,GT_list,predict_list = test(model,test_dataloader,device,0,1)

    cm = confusion_matrix(y_true=GT_list, y_pred=predict_list, normalize='true')
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4]).plot(cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"result/cm_{model_name}.png")
    plt.close()

if __name__ == '__main__':
    args = parse_args()

    if args.model == "resnet18":
        model = ResNet18()
    elif args.model == "resnet50":
        model = ResNet50()
    elif args.model == "resnet18_pretrain":
        model = ResNet18_pretrain(5)
    elif args.model == "resnet50_pretrain":
        model = ResNet50_pretrain(5)
    
    if args.eva==1:
        weights = torch.load(f"{args.model}_weight.pth")
        model.load_state_dict(weights)
        evaluate(model,args.batch_size,"cuda",args.model)
    else:
        training(args.batch_size,args.epochs,args.lr,model,"cuda",args.model)

