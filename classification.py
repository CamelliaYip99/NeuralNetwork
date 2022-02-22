import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, clz0_num, clz1_num):
        super(Dataset, self).__init__()

        data0_size = torch.ones(clz0_num, 2)
        data0 = torch.normal(2 * data0_size, 1)
        label0 = torch.zeros(clz0_num)

        data1_size = torch.ones(clz1_num, 2)
        data1 = torch.normal(-2 * data1_size, 1)
        label1 = torch.ones(clz1_num)

        self.data = torch.cat((data0, data1), 0).type(torch.FloatTensor)
        self.label = torch.cat((label0, label1)).type(torch.LongTensor)

        plt.scatter(self.data.numpy()[:, 0], self.data.numpy()[:, 1], c=self.label.numpy(), s=100, lw=0,
                    cmap='RdYlGn')
        plt.show()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)


class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.linear = nn.Linear(n_input, 10)
        self.relu = F.relu
        self.output = nn.Linear(10, n_output)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.output(x)
        return x


clz0_num = 100
clz1_num = 100

dataset = Dataset(clz0_num, clz1_num)

data_loader = data.DataLoader(dataset=dataset, batch_size=200, shuffle=False, num_workers=6)

net = Net(n_input=2, n_output=2).cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_function = torch.nn.CrossEntropyLoss()

losses = []

for epoch in range(200):
    for _, train_data in enumerate(data_loader):
        data = train_data[0].cuda()
        label = train_data[1].cuda()

        pred = net(data)

        loss = loss_function(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)

        if epoch % 10 == 0:
            pred_label = torch.max(pred, 1)[1]
            pred_label = pred_label.cpu().detach().numpy()
            target_label = label.cpu().numpy()
            plt.scatter(data.cpu().numpy()[:, 0], data.cpu().numpy()[:, 1], c=pred_label, s=100, lw=0, cmap='RdYlGn')
            accuracy = float((pred_label == target_label).astype(int).sum()) / float(target_label.size)
            plt.text(1.5, -4, 'Acc.=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.show()

batch_nums = range(1, len(losses) + 1)
plt.plot(batch_nums, losses)
plt.title('Loss - Batch')
plt.xlabel('batch')
plt.ylabel('loss')
plt.show()
