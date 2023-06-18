import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
#from models.densenet import DenseNet3
from models.wrn import WideResNet

from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import logging
# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance_log, get_measures, print_measures_log, print_measures_with_std_log
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    import utils.score_calculation as lib
    from utils.SCOODBenchmarkDataset import SCOODDataset
parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_zallconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos',default='true', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--gamma1', type=float, default=0.0, help='influence_gamma1:for dist')
parser.add_argument('--gamma2', type=float, default=0.0, help='influence_gamma2:for loss')
parser.add_argument('--trial', type=int, default=1, help='trial one, two three')

args = parser.parse_args()
print(args)
# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviat
# ion of channels of CIFAR-10 images


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

#test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
#test_transform = trn.Compose([trn.ToTensor()])
test_transform=trn.Compose([trn.Resize((32,32)),trn.ToTensor(), trn.Normalize(mean, std)])
if 'cifar10_' in args.method_name:
    data_set='cifar10'
    test_data = dset.CIFAR10('./data/cifar10', train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100('./data/cifar100', train=False, transform=test_transform)
    num_classes = 100
    data_set='cifar100'


test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)


start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'pretrained' in args.method_name:
            subdir = 'pretrained'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        elif 'energy_ft' in args.method_name:
            subdir = 'energy_ft'
        else:
            subdir = 'oe_scratch'
        
        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i)+'_gamma_'+str(args.gamma1)+ str(args.gamma2)+'_trial_'+str(args.trial)  + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume "+model_name

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename=model_name+'.txt')  # pass explicit filename here


net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data)  ####33all data
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores_SCOOD(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []
    _sc_label=[]
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):


            data = data.cuda()
            target=target.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            _sc_label.append(target.detach().cpu().numpy())
            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
                else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
        ood_scores=np.concatenate(_score, axis=0)
        sc_labels=np.concatenate(_sc_label,axis=0)
        fake_ood_scores=ood_scores[sc_labels>=0]
        real_ood_scores=ood_scores[sc_labels<0]
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:

        return ood_scores[:ood_num_examples].copy(), fake_ood_scores, real_ood_scores

def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
                else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()
if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, ood_num_examples, args.T, args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable
    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)


    if 'cifar10_' in args.method_name:
        train_data = dset.CIFAR10('./data/cifar10', train=True, transform=test_transform)
    else:
        train_data = dset.CIFAR100('./data/cifar100', train=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False, 
                                          num_workers=args.prefetch, pin_memory=True)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader) 
    in_score = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])
else:
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
logger.info('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')
logger.info('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
logger.info('\n\nError Detection')

#show_performance(wrong_score, right_score, method_name=args.method_name)
show_performance_log(logger, wrong_score, right_score, method_name=args.method_name)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(logger, ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches)
        else:
            out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        #print_measures_with_std(aurocs, auprs, fprs, args.method_name)
        print_measures_with_std_log(logger, aurocs, auprs, fprs, args.method_name)

    else:
        #print_measures(auroc, aupr, fpr, args.method_name)
        print_measures_log(logger, auroc, aupr, fpr, args.method_name)


def get_and_print_results_SCOOD(logger, ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches)
        else:
            out_score, fake_ood_scores, real_ood_scores = get_ood_scores_SCOOD(ood_loader)
            real_in_scores=np.concatenate([in_score,fake_ood_scores], axis=0)
            if args.out_as_pos: # OE's defines out samples as positive
                measures = get_measures(real_ood_scores, real_in_scores)
                #print(real_ood_scores.size(), real_in_scores.size())
            else:
                measures = get_measures(-real_in_scores, -real_ood_scores)
                #print(real_ood_scores.shape, real_in_scores.shape)                
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        #print_measures_with_std(aurocs, auprs, fprs, args.method_name)
        print_measures_with_std_log(logger, aurocs, auprs, fprs, args.method_name)

    else:
        #print_measures(auroc, aupr, fpr, args.method_name)
        print_measures_log(logger, auroc, aupr, fpr, args.method_name)
###////////////////////SCOOOD //////////////////#########33

SCOOD_dataset=['texture','svhn','cifar','tin','lsun', 'places365']
test_transform_SCOOD=trn.Compose([trn.Resize((32,32)),trn.ToTensor(), trn.Normalize(mean, std)])
#test_transform_SCOOD=trn.Compose([trn.Resize((32,32)),trn.ToTensor()])
for dout in SCOOD_dataset:
    if dout=='cifar':
        if data_set=='cifar10':
            dout='cifar100'
        elif data_set=='cifar100':
            dout='cifar10'
    ood_data= SCOODDataset(root='./data', id_name=data_set, ood_name=dout, transform=test_transform_SCOOD)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=False,
                                             num_workers=4, drop_last=False, pin_memory=True)
    print(f'\n\n SCOOD {dout} Detection')
    logger.info(f'\n\n SCOOD {dout} Detection')
    get_and_print_results_SCOOD(logger,ood_loader)

