import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split


def make_dataset(version=None, test=False):
    if test:
        random_state = None
    else:
        random_states = [27,33,38]
        if version is None:
            version = random.choice(range(len(random_states)))
            print(f"Dataset number: {version}")
        random_state = random_states[version]
    return sklearn.datasets.make_circles(factor=0.7, noise=0.1, random_state=random_state)


import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()

        self.linear = nn.Linear(n_input, 1000)
        self.linear_1 = nn.Linear(1000, 1000)
        self.relu = F.relu
        self.output = nn.Linear(1000, n_output)
        
    def forward(self, x):

        #x = self.pool(F.relu(self.conv(x)))
        x = self.linear(x)
        x = self.relu(x)
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_1(x))
        x = self.output(x)
        return x
  
    
net = Net(n_input=2, n_output=2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_function = torch.nn.CrossEntropyLoss()
losses = []


for x in range(3):
    train_set = make_dataset(version = x, test = False)
    test_set = make_dataset(version = x, test = True)
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    
    for epoch in range(1500):
        data = torch.tensor(X_train,dtype=torch.float)
        label = torch.tensor(y_train,dtype=torch.float)

        pred = net(data)

        label = label.type(torch.LongTensor)
        loss = loss_function(pred,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)

        if epoch % 10 == 0:
            pred_label = torch.max(pred, 1)[1]
            pred_label = pred_label.cpu().detach().numpy()
            target_label = label.cpu().numpy()

            plt.scatter(data.cpu().numpy()[:, 0], data.cpu().numpy()[:, 1], c=pred_label, s=100, lw=0, cmap='RdYlGn')
            #accuracy = roc_auc_score(target_label, pred_label)
            accuracy = float((pred_label == target_label).astype(int).sum()) / float(target_label.size)
            plt.text(1, -2, 'Acc.=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.show()
            
            
pred_label = net(torch.tensor(X_test,dtype = torch.float))
pred_label = torch.max(y_pred, 1)[1]
y_pred = pred_label.detach().numpy()


data = torch.tensor(X_test)
target = torch.tensor(y_test)
target_label = label.type(torch.LongTensor)


accuracy = float((y_pred == y_test).astype(int).sum()) / float(target_label.size)


plt.scatter(data.cpu().numpy()[:, 0], data.cpu().numpy()[:, 1], c=(y_pred == y_test), s=100, lw=0, cmap='RdYlGn')
plt.text(1, -2, 'Acc.=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
plt.show()



plt.scatter(data.cpu().numpy()[:, 0], data.cpu().numpy()[:, 1], c=target_label, s=100, lw=0)
plt.text(1, -2, 'Acc.=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
plt.show()


mins = []
maxs = []
mins.append(data.numpy()[:,0].min() - 0.1) 
mins.append(data.numpy()[:,1].min() - 0.1) 
maxs.append(data.numpy()[:,0].max() + 0.1)
maxs.append(data.numpy()[:,1].max() + 0.1)
'''mins = X.numpy()[:, ].min() - 0.1
maxs = X.numpy()[:, ].max() + 0.1'''
x1grid = np.arange(mins[0], maxs[0], 0.1)
x2grid = np.arange(mins[1], maxs[1], 0.1)

xx, yy = np.meshgrid(x1grid ,x2grid)
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1,r2))

y_grid = net(torch.tensor(grid,dtype=torch.float))
y_grid = torch.max(y_grid, 1)[1]

zz = y_grid.reshape(xx.shape)


plt.contourf(xx,yy,zz.detach().numpy(),cmap="Pastel2")
plt.scatter(data.cpu().numpy()[:, 0], data.cpu().numpy()[:, 1], c=target_label)


from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 


svm_clf = SVC(kernel='rbf',gamma='auto')
svm_clf.fit(X_train, y_train)
q_estimate_svm = svm_clf.predict(X_test)

svm_auc_score = roc_auc_score(y_test, q_estimate_svm)*100
print('Training AUC: %.4f %%' % svm_auc_score)
