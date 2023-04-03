import matplotlib.pyplot as plt
import numpy as np

class Record():
    def __init__(self,model_name) -> None:
        self.model_name = model_name
        self.train_acc = {}
        self.train_L = {}
        self.test_acc = {}

    def train_accuracy(self,correct_num,total_num,activate):
        if activate not in self.train_acc:
            self.train_acc[activate] = []

        accuracy = 100. * correct_num/total_num

        self.train_acc[activate].append(accuracy)

        return accuracy


    def test_accuracy(self,correct_num,total_num,activate):
        if activate not in self.test_acc:
            self.test_acc[activate] = []

        accuracy = 100. * correct_num/total_num

        self.test_acc[activate].append(accuracy)

        return accuracy

    def plot_acc(self,epochs):
        x = np.arange(0,epochs,1)
        
        fig, ax = plt.subplots()

        for key in self.train_acc:
            ax.plot(x,self.train_acc[key],label = "train-"+key)

        for key in self.test_acc:
            ax.plot(x,self.test_acc[key],label = "test-"+key)
        
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy(%)")
        ax.set_title(f"Activation function comparison({self.model_name})")

        plt.savefig(f"plot_{self.model_name}.png")

