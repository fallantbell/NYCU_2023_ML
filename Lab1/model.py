import numpy as np

class MLP():
    def __init__(self,hidden_layer=2,hidden_size=100,activate_funtion="sigmoid"):
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.input_size = 2
        self.output_size = 1
        self.activate_function = activate_funtion

        self.weight = {}
        self.out = {}
        self.gradient = {}
        self.build()
    
    def build(self):
        
        for i in range(self.hidden_layer+1):
            if i==0:
                self.weight[f'B{i+1}'] = np.zeros((1,self.hidden_size))
                self.weight[f'W{i+1}'] = np.random.randn(self.input_size,self.hidden_size)
            elif i==self.hidden_layer:
                self.weight[f'B{i+1}'] = np.zeros((1,self.output_size))
                self.weight[f'W{i+1}'] = np.random.randn(self.hidden_size,self.output_size)
            else:
                self.weight[f'B{i+1}'] = np.zeros((1,self.hidden_size))
                self.weight[f'W{i+1}'] = np.random.randn(self.hidden_size,self.hidden_size)
        

    def loss(self,predict,GT):
        L = np.square(np.subtract(predict,GT)).mean()
        return L

    def activate(self,x,af):
        if af=="sigmoid":
            return 1.0/(1.0+np.exp(-x))
        if af=="relu":
            return np.maximum(0,x)
    def derivative_activate(self,x,af):
        if af=="sigmoid":
            return np.multiply(x,1.0-x)
        if af=="relu":
            dr = np.zeros_like(x)
            dr[x>0] = 1
            return dr
            
        
    def backward(self,predict,y):
        dL = predict-y
        for i in range(self.hidden_layer+1,0,-1):
            if i== self.hidden_layer+1:
                dL = self.derivative_activate(self.out[f'{i}'],"sigmoid")*dL
            else:
                dL = self.derivative_activate(self.out[f'{i}'],self.activate_function)*dL
               
            self.gradient[f'B{i}'] = np.sum(dL,axis=0)
            self.gradient[f'W{i}'] = np.matmul(self.out[f'{i-1}'].T,dL)
            dL = np.matmul(dL,self.weight[f'W{i}'].T)
    
    def update_weight(self,lr = 0.001):
        for i in range(1,self.hidden_layer+2,1):
            self.weight[f'W{i}'] = self.weight[f'W{i}'] - lr*self.gradient[f'W{i}']
            self.weight[f'B{i}'] = self.weight[f'B{i}'] - lr*self.gradient[f'B{i}']


    def forward(self,x):

        self.out[f'{0}']=x
        for i in range (self.hidden_layer+1):
            x = np.matmul(x,self.weight[f'W{i+1}'])
            x = x + self.weight[f'B{i+1}']     
            if i==self.hidden_layer:
                x = self.activate(x,"sigmoid")
            else:
                x = self.activate(x,self.activate_function)
            self.out[f'{i+1}']=x
        
        return x


    def accuracy(self,predict,y):
        predict2 = np.copy(predict)
        predict2[predict2 > 0.5] = 1
        predict2[predict2 <= 0.5] = 0

        return 100 * np.sum(predict2 == y)/predict2.shape[0]

