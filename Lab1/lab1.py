import numpy as np
from generate import Generate_dataset
import matplotlib.pyplot as plt
from model import MLP
    

def train_and_test(mlp,G,data="linear"):
    if data == "linear":
        x, y = G.generate_linear(n=100)
    else:
        x, y = G.generate_XOR_easy()

    epoch = 0
    lr = 0.01
    loss_record = []

    print(f"\n ------ {data} traing ------ \n")

    while True:
        epoch += 1

        predict = mlp.forward(x)

        L = mlp.loss(predict,y)

        loss_record.append(L)

        accuracy = mlp.accuracy(predict,y)

        if epoch % 100 ==0:
            print(f"epoch {epoch}: loss = {L} , accuracy = {accuracy} %")

        if accuracy >= 100:##
            print(f"epoch {epoch}: loss = {L} , accuracy = {accuracy} %")
            break
        if epoch > 30000:
            print("something wrong")
            break 
        
        mlp.backward(predict,y)
        mlp.update_weight(lr = lr)

    pred_y = mlp.forward(x)
    print(f"result:")
    print(f"{pred_y}")
    print(" ")

    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0
    accuracy = mlp.accuracy(pred_y,y)
    print(f"accuracy: {accuracy}")
    print(" ")

    G.show_result(x,y,f"{data}.png",pred_y)

    plt.plot(loss_record)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(f"{data}_loss.png")
    plt.close()

if __name__ == '__main__':

    mlp = MLP(hidden_layer = 2,
          hidden_size = 10,
          activate_funtion="sigmoid") 
    
    G = Generate_dataset()   
            
    train_and_test(mlp,G,data="linear")

    mlp2 = MLP(hidden_layer = 2,
          hidden_size = 10,
          activate_funtion="sigmoid")  

    G2 = Generate_dataset()   
    train_and_test(mlp2,G2,data="xor")


