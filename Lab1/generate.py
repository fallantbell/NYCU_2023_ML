import numpy as np
import matplotlib.pyplot as plt

class Generate_dataset():
    def __init__(self):
        pass

    def generate_linear(self,n=100):
        import numpy as np
        pts = np.random.uniform(0,1,(n,2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0],pt[1]])
            distance = (pt[0]-pt[1])/1.414
            if pt[0]>pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n,1)
    
    def generate_XOR_easy(self):
        inputs = []
        labels = []

        for i in range(11):
            inputs.append([0.1*i,0.1*i])
            labels.append(0)

            if 0.1*i == 0.5:
                continue

            inputs.append([0.1*i,1-0.1*i])
            labels.append(1)
        return np.array(inputs), np.array(labels).reshape(21,1)
    
    def show_result(self,x,y,path,pred_y=0):
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.title('Ground truth',fontsize = 18)
        for i in range(x.shape[0]):
            if y[i]==0:
                plt.plot(x[i][0],x[i][1],'ro')
            else:
                plt.plot(x[i][0],x[i][1],'bo')
        plt.subplot(1,2,2)
        plt.title('Predict result',fontsize = 18)
        for i in range(x.shape[0]):
            if pred_y[i]==0:
                plt.plot(x[i][0],x[i][1],'ro')
            else:
                plt.plot(x[i][0],x[i][1],'bo')
        plt.savefig(f'{path}')
        plt.close()