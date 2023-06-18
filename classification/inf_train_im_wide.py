# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet
from torch.utils.data import DataLoader, Subset
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.tinyimages_300k import TinyImages
    from utils.validation_dataset import validation_split
    from utils.ImbalanceCIFAR import IMBALANCECIFAR10
    from utils.ImbalanceCIFAR import IMBALANCECIFAR100
parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='allconv', help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/pretrained', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='OE', help='OE|energy')
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--num_ood_samples', default=30000, type=float, help='Number of OOD samples to use.')
parser.add_argument('--imbalance_ratio', type=float, default=0.01, help='Imbalance ratio small is big imbal')

parser.add_argument('--lamb', type=float, default=5.0, help='influence_dist_lamb')
parser.add_argument('--gamma1', type=float, default=0.0, help='influence_gamma1:for dist')
parser.add_argument('--gamma2', type=float, default=0.0, help='influence_gamma2:for loss')

parser.add_argument('--trial', type=int, default=1, help='trial one, two three')

args = parser.parse_args()


if args.score == 'OE':
    save_info = 'oe_tune'
elif args.score == 'energy':
    save_info = 'energy_ft'

args.save = args.save+save_info
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
#test_transform = trn.Compose([trn.ToTensor()])
if args.dataset == 'cifar10':
    train_data_in= IMBALANCECIFAR10(phase="train", imbalance_ratio=args.imbalance_ratio,root = './data/cifar10',imb_type='exp')
    test_data = dset.CIFAR10('./data/cifar10', train=False, transform=test_transform)
    num_classes = 10
    prior_gamma1=torch.tensor([0.313627,0.07421,0.157883,0.230267,0.047117,0.079,0.026387,0.024573,0.014643,0.032293]).cuda()**args.gamma1
    prior_gamma2=torch.tensor([0.313627,0.07421,0.157883,0.230267,0.047117,0.079,0.026387,0.024573,0.014643,0.032293]).cuda()**args.gamma2
    prior_gamma1=F.normalize(prior_gamma1,p=1.0, dim=0)
    prior_gamma2=F.normalize(prior_gamma2,p=1.0, dim=0)
    prior_gamma1=prior_gamma1[None,:]
    prior_gamma2=prior_gamma2[None,:]

else:
    train_data_in= IMBALANCECIFAR100(phase="train", imbalance_ratio=args.imbalance_ratio,root = './data/cifar100',imb_type='exp')
    test_data = dset.CIFAR100('./data/cifar100', train=False, transform=test_transform, download=True)
    num_classes = 100
    prior_gamma1=torch.tensor([0.005863,0.01383,0.06535,0.02444,0.012077,0.048303,0.014427,0.009493,0.01728,0.045707,0.035927,0.059217,0.035387,0.046853,0.010873,0.013507,0.061203,0.023257,0.02249,0.019043,0.014443,0.009377,0.032313,0.022807,0.003373,0.03525,0.01097,0.011733,0.008137,0.005393,0.006173,0.005653,0.01175,0.024213,0.0028,0.01844,0.002977,0.020523,0.006067,0.019587,0.016667,0.007007,0.008227,0.00185,0.005953,0.008703,0.021377,0.004473,0.003637,0.005713,0.001303,0.003657,0.00119,0.001453,0.006067,0.00056,0.001873,0.00108,0.002143,0.002943,0.008353,0.00376,0.00142,0.000707,0.00121,0.00046,0.001083,0.0012,0.002717,0.00258,0.002267,0.002917,0.00029,0.0006,1.67E-05,0.000393,0.001633,5.33E-05,0.00012,0.00002,3.33E-06,6.67E-05,0.000653,0.00015,0.00134,0.000217,0.000253,0.000347,0.000153,2.67E-05,0.000223,0,0,3.33E-06,0.002333,0,0,0,4.33E-05,3.33E-06]).cuda()**args.gamma1
    prior_gamma2=torch.tensor([0.005863,0.01383,0.06535,0.02444,0.012077,0.048303,0.014427,0.009493,0.01728,0.045707,0.035927,0.059217,0.035387,0.046853,0.010873,0.013507,0.061203,0.023257,0.02249,0.019043,0.014443,0.009377,0.032313,0.022807,0.003373,0.03525,0.01097,0.011733,0.008137,0.005393,0.006173,0.005653,0.01175,0.024213,0.0028,0.01844,0.002977,0.020523,0.006067,0.019587,0.016667,0.007007,0.008227,0.00185,0.005953,0.008703,0.021377,0.004473,0.003637,0.005713,0.001303,0.003657,0.00119,0.001453,0.006067,0.00056,0.001873,0.00108,0.002143,0.002943,0.008353,0.00376,0.00142,0.000707,0.00121,0.00046,0.001083,0.0012,0.002717,0.00258,0.002267,0.002917,0.00029,0.0006,1.67E-05,0.000393,0.001633,5.33E-05,0.00012,0.00002,3.33E-06,6.67E-05,0.000653,0.00015,0.00134,0.000217,0.000253,0.000347,0.000153,2.67E-05,0.000223,0,0,3.33E-06,0.002333,0,0,0,4.33E-05,3.33E-06]).cuda()**args.gamma2
    prior_gamma1=F.normalize(prior_gamma1,p=1.0, dim=0)
    prior_gamma2=F.normalize(prior_gamma2,p=1.0, dim=0)
    prior_gamma1=prior_gamma1[None,:]
    prior_gamma2=prior_gamma2[None,:]

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

#ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform), list(range(args.num_ood_samples)))

ood_data = Subset(TinyImages(root='./data', transform=trn.Compose(
    [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
     trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)])), list(range(args.num_ood_samples)))
ood_data_inf=TinyImages(root='./data', transform=test_transform)

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.oe_batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

test_loader_out = torch.utils.data.DataLoader(
    ood_data_inf,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)
# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module
# Restore model
model_found = False
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        
        model_name = os.path.join(args.load, args.dataset + calib_indicator + '_' + args.model +
                                  '_pretrained_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////
def influence(value, weight, dim=None, keepdim=False):

    influence= torch.sum(value*F.relu(weight), dim=dim, keepdim=keepdim)
    #influence = influence.squeeze(dim)

    return influence

def train():
    net.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        softmax=F.softmax(x,dim=1)
        influence_gamma1=influence(softmax,prior_gamma1,dim=1)
        influence_gamma2=influence(softmax,prior_gamma2,dim=1)

        Ec_out_dist=influence_gamma1[len(in_set[0]):]
        Ec_out_weight=influence_gamma2[len(in_set[0]):]
        if args.score == 'energy':
            Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
            Ec_out = Ec_out - (args.lamb * Ec_out_dist)
            Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
            loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + (torch.pow(F.relu((args.m_out - Ec_out)), 2)*Ec_out_weight).sum()/Ec_out_weight.sum())
        elif args.score == 'OE':
            loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)



if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info + '_gamma_' + str(args.gamma1)+ str(args.gamma2) + '_trial_' + str(
    args.trial) + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch) + '_gamma_' + str(args.gamma1)+ str(args.gamma2) + '_trial_' + str(
                   args.trial) + '.pt'))

    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                             '_' + save_info + '_epoch_' + str(epoch - 1) + '_gamma_' + str(
        args.gamma1)+ str(args.gamma2) + '_trial_' + str(args.trial) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results
    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                      '_' + save_info + '_gamma_' + str(args.gamma1)+ str(args.gamma2) + '_trial_' + str(
        args.trial) + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
