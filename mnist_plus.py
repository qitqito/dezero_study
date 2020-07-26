import dezero
import dezero.functions as F
from dezero import DataLoader
from dezero.models import Model
import dezero.layers as L


class MNISTPlus(Model):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.conv1 = L.Conv2d(30, kernel_size=3, stride=1, pad=1)
        #self.conv2 = L.Conv2d(1, kernel_size=3, stride=1, pad=1)
        self.fc3 = L.Linear(hidden_size)
        #self.fc4 = L.Linear(hidden_size)
        self.fc5 = L.Linear(10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (OH, OW)=(28, 28)
        x = F.pooling(x, 2, 2) # (OH, OW)=(14, 14)
        #x = F.relu(self.conv2(x))
        #x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1)) # (14, 14)->(196, )
        x = F.dropout(F.relu(self.fc3(x)))
        #x = F.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

max_epoch = 20
batch_size = 100

train_set = dezero.datasets.MNIST(train=True, transform=None) # (28, 28)
test_set = dezero.datasets.MNIST(train=False, transform=None) # (28, 28)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MNISTConv()
optimizer = dezero.optimizers.Adam().setup(model)
optimizer.add_hook(dezero.optimizers.WeightDecay(1e-4))  # Weight decay

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    test_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {}, accuracy: {}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {}, accuracy: {}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))