import numpy as np
l=np.array([])
def sigmoid(value):
    return 1/1+np.exp(-value)

def perceptron():
    arr = np.array([list(map(float,input().split()))])
    size= np.shape(arr)
    weight=np.random.rand(1,size[1])
    bias = np.random.rand()
    value = (np.dot(arr,weight[0]))+bias
    l=np.array([sigmoid(value)])


if  __name__=='__main__':
    n=int(input("Enter number of perceptrons:"))
    for i in range(n):
        perceptron()
    print(l)
