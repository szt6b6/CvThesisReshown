from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import sys
import os
import time

def load_minist_data(data_dir="ministDataset", batch_size=256):

    # 数据预处理：1. 将输入图片的大小宽展为 32x32；2. 进行标准化，转化为均值为 0，方差为 1 的数据
    # transforms.Normalize((0.1307,), (0.3081,)) 的讨论参见：
    #       https://discuss.pytorch.org/t/normalization-in-the-mnist-example
    data_transform = transforms.Compose([transforms.Pad(2),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=data_transform)
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = datasets.MNIST(data_dir, train=False, transform=data_transform)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader
def load_minist_fashion_data(data_dir="ministDataset", batch_size=256):

    # 数据预处理：1. 将输入图片的大小宽展为 32x32；2. 进行标准化，转化为均值为 0，方差为 1 的数据
    # transforms.Normalize((0.1307,), (0.3081,)) 的讨论参见：
    #       https://discuss.pytorch.org/t/normalization-in-the-mnist-example
    data_transform = transforms.Compose([transforms.Pad(2),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=data_transform)
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = datasets.FashionMNIST(data_dir, train=False, transform=data_transform)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader



def train(train_data, test_data, net, loss_func, epochs, learning_rate):
        #收集训练时的损失和精度 并记录写如文件中
        logspath = "logs.csv"
        if(not os.path.exists(logspath)):
             os.system("echo epoch,time,model_name,train_loss,train_acc,test_acc> %s" % logspath)
        f = open(logspath, "a")

        optimizer = optim.Adam(net.parameters(), learning_rate)

        for epoch in range(epochs):
            train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
            for x, y in train_data:
                x = x.to(device)
                y = y.to(device)

                y_hat = net(x)
                loss = loss_func(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            
            print("epoch %d, loss: %.4f, train acc: %.3f" % (epoch, train_loss_sum, train_acc_sum / n))

            # test mode
            test_acc_sum, n_test = 0, 0
            with torch.no_grad():
                net.eval() #评估模式 会关闭dropout
                for x, y in test_data:
                    x = x.to(device)
                    y = y.to(device)

                    y_hat = net(x)

                    test_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                    n_test += y.shape[0]
                
                net.train()#改回训练模式
            print("test acc: %.3f" %(test_acc_sum / n_test))

            f.writelines("%d, %s, %s, %.4f, %.3f, %.3f\n" % (\
                 epoch, time.strftime("%Y-%m-%d %H:%M:%S"), net._get_name(), train_loss_sum, train_acc_sum / n, test_acc_sum / n_test))
        f.close()

if __name__ == '__main__':
    #读入数据
    train_data_loader, test_data_loader = load_minist_fashion_data()
    #查看cuda是是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    #导入与创建网络
    sys.path.append('models')
    # from models.Lenet import Lenet
    # net = Lenet()
    # from models.SENet import SE_LEnet
    # net = SE_LEnet(r=4)
    # from models.ResNet import ResNetForMinist
    # net = ResNetForMinist()

    from models.Vit_transformer import Vit_Transformer
    net = Vit_Transformer(1, 32, 64, 32, 4)
    
    net.to(device)
    print("train on " + str(device))

    #定义epoch，学习率， 损失， 默认使用Adam方法反向传播
    epochs = 10
    lr = 0.001
    loss_func = torch.nn.CrossEntropyLoss()

    

    time1 = time.time()
    train(train_data=train_data_loader, test_data=test_data_loader, 
          net=net, loss_func=loss_func, epochs=epochs, learning_rate=lr)
    print("训练耗时：%.2fs" % (time.time() - time1))
    # 保存模型
    torch.save(net, "pre_trained\\" + net._get_name() + ".pth")
    print("模型保存为: " + "pre_trained\\" + net._get_name() + ".pth")

