import torch as T


def print_cuda():
    dev = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
    print(dev)
    print(T.cuda.is_available())

if __name__ == "__main__":
    dev = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
    print(dev)
    print(T.cuda.is_available())