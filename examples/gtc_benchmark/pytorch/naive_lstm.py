'''
Author: Qipeng Guo
'''


from time import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class NaiveLSTM(nn.Module):
    def __init__(self, num_hidden, patch_size):
        super(NaiveLSTM, self).__init__()

        self._num_hidden = num_hidden
        self._patch_size = patch_size

        self._xi = nn.Linear(patch_size, num_hidden)
        self._xf = nn.Linear(patch_size, num_hidden)
        self._xo = nn.Linear(patch_size, num_hidden)
        self._xg = nn.Linear(patch_size, num_hidden)

        self._hi = nn.Linear(num_hidden, num_hidden)
        self._hf = nn.Linear(num_hidden, num_hidden)
        self._ho = nn.Linear(num_hidden, num_hidden)
        self._hg = nn.Linear(num_hidden, num_hidden)

        self._linear = nn.Linear(num_hidden, 10)

    def _step(self, x, h, c):

        i = F.sigmoid(self._xi(x) + self._hi(h))
        f = F.sigmoid(self._xf(x) + self._hf(h))
        o = F.sigmoid(self._xo(x) + self._ho(h))
        g = F.tanh(self._xg(x) + self._hg(h))

        c = f * c + i * g
        h = o * F.tanh(c)

        return h, c

    def forward(self, data):
        #data = data.view(data.size(0), -1, self._patch_size)
        N, L= data.size(0), data.size(1)

        h = Variable(torch.zeros(N, self._num_hidden).float().cuda(), requires_grad=False)
        c = Variable(torch.zeros(N, self._num_hidden).float().cuda(), requires_grad=False)

        #print data[:,0].size(), h.size(), c.size()

        for i in range(L):
            patch = data[:, i]
            h, c = self._step(patch, h, c)

        return F.log_softmax(self._linear(h))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_hidden', type=int, required=True)
    parser.add_argument('--patch', type=int, default=7)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('./data', train=True, download=True, 
                      transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                      ])),
      batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('./data', train=False, 
                      transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                      ])),
      batch_size=args.batch_size, shuffle=True, **kwargs)    
    

    model = NaiveLSTM(args.num_hidden, args.patch)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    tft = 0 # training forward
    ift = 0 # inference forward
    bt = 0 # backward

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data.resize_(data.size(0), 784/args.patch, args.patch)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        
        t0 = time()
        output = model(data)
        tft += time() - t0

        # ignore time difference caused by nll loss implementation
        loss = F.nll_loss(output, target) 
        
        t0 = time()
        loss.backward()
        bt += time() - t0
        optimizer.step()
        
        if (batch_idx + 1) % 20 == 0:
            print 'tft', tft / 20, 'bt', bt / 20
            if (batch_idx + 1) == 100: break
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                0, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            '''
            
    tft /= (batch_idx + 1)
    bt /= (batch_idx + 1)               
    print tft, bt

    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data.resize_(data.size(0), 784/args.patch, args.patch)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        
        t0 = time()
        output = model(data)
        ift += time() - t0
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

        if (batch_idx + 1) == 100: break

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    '''
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    '''

    ift /= (batch_idx + 1)
    print ift

    import cPickle as pickle
    identifier = 'time/naive-lstm-batch-%d-hidden-%d' % (args.batch_size, args.num_hidden)
    pickle.dump((tft, ift, bt,), open(identifier, 'w'))
