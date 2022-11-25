import data
import torch
from torch.nn import Module, Linear, ReLU, LogSoftmax, CrossEntropyLoss
from torch.optim import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class MLP(Module):
    """
    Neural Network Classifier
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.fc1 = Linear(self.input_dim, self.hidden_dim1)
        self.fc2 = Linear(self.hidden_dim1, self.hidden_dim2)
        self.output = Linear(self.hidden_dim2, output_dim)

        self.relu = ReLU()
        self.softmax = LogSoftmax()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.output(out)
        return out

    def predict(self, x):
        x = torch.FloatTensor(x)
        out = self.forward(x).detach().numpy()
        out = np.where(out == np.max(out), 1, 0)
        return out 

def test(net, X_train, y_train, X_test, y_test):

    test_out = np.argmax(net(torch.FloatTensor(X_test)).detach().numpy(), 1)
    train_out = np.argmax(net(torch.FloatTensor(X_train)).detach().numpy(), 1)

    # test_accuracy = np.sum(np.where(test_out == y_test, 1, 0))/y_test.shape[0]
    # train_accuracy = np.sum(np.where(train_out == y_train, 1, 0))/y_train.shape[0]

    acc1 = metrics.accuracy_score(y_pred=test_out, y_true=y_test)
    acc2 = metrics.accuracy_score(y_pred=train_out, y_true=y_train)

    print('Train accuracy: ' + str(acc1))
    print('Test accuracy: ' + str(acc1))

def train(net, X_train, y_train, X_test, y_test, batch_size=10, epochs=200, lr=0.0005):

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)

    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=5e-8)

    cost = {'loss': [], 'epoch': []}
    
    for i in range(epochs):

        batch_loss = 0
        randShuffle = torch.randperm(X_train.size()[0])
        X_train_shuff = X_train[randShuffle]
        y_train_shuff = y_train[randShuffle]        

        for j in range(0, X_train.size()[0], batch_size):

            X_batch = X_train_shuff[j:j+batch_size]
            y_batch = y_train_shuff[j:j+batch_size]

            optimizer.zero_grad()

            out = net(X_batch)

            loss = criterion(out.squeeze(), y_batch)   

            batch_loss += loss   
            
            loss.backward()
            optimizer.step()

        print(batch_loss)

        cost['loss'].append(batch_loss/(X_train.size()[0]/batch_size))
        cost['epoch'].append(i+1)

        test(net, X_train, y_train, X_test, y_test)
        test(net, X_train_shuff, y_train_shuff, X_test, y_test)

    plt.plot(cost['epoch'], cost['loss'])
    plt.show()

    return net

def main():

    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    num_classes = 10
    net = MLP(64, 16, 16, num_classes)
    model = train(net, X_train=train_data, y_train=train_labels, X_test=test_data, y_test=test_labels)

    test(model, train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    main()
