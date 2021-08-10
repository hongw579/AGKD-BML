'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import numpy as np
import random
import math
import os
from argument import parser, print_args, create_logger

from tqdm import tqdm
from custom_models import *
from utils import *

# Model
class Attacks(nn.Module):
    def __init__(self, basic_net, config):
        super(Attacks, self).__init__()
        self.basic_net = basic_net
        self.epsilon_PGD = config['epsilon_PGD']
        self.num_steps_PGD = config['num_steps_PGD']
        self.step_size_PGD = config['step_size_PGD']
        self.epsilon_MC = config['epsilon_MC']
        self.num_steps_MC = config['num_steps_MC']
        self.step_size_MC = config['step_size_MC']
        self.rand = config['random_start']

        assert config['loss_func'] == 'xent', 'Only xent supported for now.'

    def perturb_PGD(self, inputs, targets, targeted = False):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon_PGD, self.epsilon_PGD)
        self.basic_net.eval()
        for i in range(self.num_steps_PGD):
            x.requires_grad_()
            with torch.enable_grad():
                _, _, logits = self.basic_net(x, _eval = True)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            if targeted:
                x = x.detach() - self.step_size_PGD*torch.sign(grad.detach())
            else:
                x = x.detach() + self.step_size_PGD*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon_PGD), inputs + self.epsilon_PGD)
            x = torch.clamp(x, -1.0, 1.0)
        self.basic_net.train()
        return x

    def targeted_MC(self, inputs, targets):
        x = inputs.detach()
        _, _, nat_logits = self.basic_net(x, _eval = True)
        MC_targets = MC_labels(nat_logits, targets)
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon_MC, self.epsilon_MC)
        self.basic_net.eval()
        for i in range(self.num_steps_MC):
            x.requires_grad_()
            with torch.enable_grad():
                _, _, logits = self.basic_net(x, _eval = True)       
                loss = F.cross_entropy(logits, MC_targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() - self.step_size_MC*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon_MC), inputs + self.epsilon_MC)
            x = torch.clamp(x, -1.0, 1.0)
        self.basic_net.train()
        return x, MC_targets

    def perturb_CW(self, inputs, targets):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon_PGD, self.epsilon_PGD)
        self.basic_net.eval()
        for i in range(self.num_steps_PGD):
            x.requires_grad_()
            with torch.enable_grad():
                _, _, logits = self.basic_net(x, _eval = True)
                loss = CW_loss(logits, targets)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size_PGD*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon_PGD), inputs + self.epsilon_PGD)
            x = torch.clamp(x, -1.0, 1.0)
        self.basic_net.train()
        return x

# Training
def train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    basic_net.train()
    train_loss = 0
    correct = 0
    nat_correct = 0
    total = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)

    for batch_idx, (input1, target1) in enumerate(iterator):
        input1, target1 = input1.to(device), target1.to(device)
        kl = nn.KLDivLoss(reduction='none')
        basic_net.eval()

        adv_input1, input1_mclabel = attack.targeted_MC(input1, target1)
        input1_mclabel = input1_mclabel.to('cpu')
        input2_idx = classes_idx.get_batch(input1_mclabel)
        input2, target2 = to_variable(trainset, input2_idx)
        input2, target2 = input2.to(device), target2.to(device)
        adv_input2 = attack.perturb_PGD(input2, target1, targeted = True)

        basic_net.train()
        input_all = torch.cat([adv_input1, input1, adv_input2], 0)
        input_nat = torch.cat([input1, input2], 0)
        features, x4s, outputs = basic_net(input_all)
        features_nat, x4s_nat, outputs_nat = nat_net(input_nat, _eval = True)

        output_all = torch.chunk(outputs, 3, dim = 0)
        x4_all = torch.chunk(x4s, 3, dim = 0)
        feature_all = torch.chunk(features, 3, dim = 0)
        output_nat_all = torch.chunk(outputs_nat, 2, dim = 0)
        feature_nat_all = torch.chunk(features_nat, 2, dim = 0)

        if config['lb_smooth']:
            loss = softmax_crossentropy_labelsmooth(output_all[0], target1, lb_smooth=config['lb_smooth'])
        else:
            loss = criterion_ori(output_all[0], target1)

        triplet_loss_apn = triplet_loss(x4_all[0], x4_all[1], x4_all[2], config['margin_A_Ap_B'], config['lam2'])
        if args.mart:
            loss_mart = mart_loss(output_all[1], output_all[0], target1, kl)
            loss = loss + loss_mart

        student_map_1 = attention_map(feature_all[0])
        teacher_map_1 = attention_map(feature_nat_all[0].detach().clone())
        loss_distillation_1 = F.l1_loss(student_map_1, teacher_map_1)
        loss = loss + loss_distillation_1 
        student_map_2 = attention_map(feature_all[2])
        teacher_map_2 = attention_map(feature_nat_all[1].detach().clone())
        loss_distillation_2 = F.l1_loss(student_map_2, teacher_map_2)
        loss = loss + loss_distillation_2 
        loss = loss + config['lam1']*triplet_loss_apn

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = output_all[0].max(1)
        _, nat_predicted = output_nat_all[0].max(1)
        total += target1.size(0)
        correct += predicted.eq(target1).sum().item()
        nat_correct += nat_predicted.eq(target1).sum().item()
        iterator.set_description(str(predicted.eq(target1).sum().item()/target1.size(0)))

    scheduler.step()
    acc = 100.*correct/total
    nat_acc = 100.*nat_correct/total

    logger.info('Train acc: %.3f' % acc)
    logger.info('Train nat acc: %.3f' % nat_acc)
    logger.info('Train loss: %.3f'% train_loss)
    logger.info('Learning Rate: %f' % get_lr(optimizer))

    state_latest = {
        'net': basic_net.state_dict(),
        'acc': acc,
        'epoch': epoch+1,
    }
    if not os.path.isdir(args.ckpt_root):
        os.mkdir(args.ckpt_root)
    train_root = os.path.join(args.ckpt_root, 'ckpt_latest.t7')
    torch.save(state_latest, train_root)

def test(epoch, loader):
    global best_acc
    logger.info('\nEpoch: %d' % epoch)
    basic_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                _, _, outputs = basic_net(inputs, _eval = True)
                loss = criterion_ori(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    # Save checkpoint.
    acc = 100.*correct/total
    logger.info('Val acc: %.3f' % acc)
    logger.info('Val loss: %.3f' % test_loss)
    if acc > best_acc:
        logger.info('Saving..')
        state = {
            'net': basic_net.state_dict(),
            'acc': acc,
            'epoch': epoch+1,
        }
        if not os.path.isdir(args.ckpt_root):
            os.mkdir(args.ckpt_root)
        test_root = os.path.join(args.ckpt_root, 'ckpt.t7')
        torch.save(state, test_root)
        best_acc = acc

def adv_test(epoch):
    global best_adv_acc
    logger.info('\nEpoch: %d' % epoch)
    basic_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                if config['adv_test_loss'] == 'ce':
                    adv_inputs = attack.perturb_PGD(inputs, targets)
                elif config['adv_test_loss'] == 'cw':
                    adv_inputs = attack.perturb_CW(inputs, targets)
                _, _, outputs = basic_net(adv_inputs, _eval = True)
                loss = criterion_ori(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    # Save checkpoint.
    acc = 100.*correct/total
    logger.info('Adv Val acc: %.3f' % acc)
    logger.info('Adv Val loss: %.3f' % test_loss)
    if acc > best_adv_acc:
        logger.info('Saving..')
        state = {
            'net': basic_net.state_dict(),
            'acc': acc,
            'epoch': epoch+1,
        }
        if not os.path.isdir(args.ckpt_root):
            os.mkdir(args.ckpt_root)
        adv_test_root = os.path.join(args.ckpt_root, 'ckpt-adv.t7')
        torch.save(state, adv_test_root)
        best_adv_acc = acc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)
    log_name = 'adv-trip'
    logger = create_logger(args.log_root, log_name, 'info')
    print_args(args, logger)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0  # best test accuracy
    best_adv_acc = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    logger.info('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # Normalization messes with l-inf bounds.
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    logger.info('==> Building model..')
    basic_net = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
    basic_net = basic_net.to(device)
    nat_net = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
    nat_net = nat_net.to(device)
    # From https://github.com/MadryLab/cifar10_challenge/blob/master/config.json
    config = {

        'margin_A_Ap_B': 0.03,
        'loss_func': 'xent',
        'epsilon_PGD': 8.0/255*2,
        'num_steps_PGD': 10,
        'step_size_PGD': 2.0/255*2,
        'epsilon_MC': 8.0/255*2,
        'num_steps_MC': 10,
        'step_size_MC': 2.0/255*2,
        'random_start': True,
        'lam1': 2,
        'lam2': 0.001,
        'lb_smooth': 0.5,
        'adv_test_loss': 'ce',
    }
    logger.info(config)
    attack = Attacks(basic_net, config)
    if device == 'cuda':
        basic_net = torch.nn.DataParallel(basic_net)
        nat_net = torch.nn.DataParallel(nat_net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        resume_dir = args.ckpt_root + '0'
        assert os.path.isdir(resume_dir), 'Error: no checkpoint directory found!'
        resume_root = os.path.join(resume_dir, 'ckpt_latest.t7')
        checkpoint = torch.load(resume_root)
        basic_net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    assert os.path.isdir(args.nat_root), 'Error: no checkpoint directory found!'
    nat_root = os.path.join(args.nat_root, args.nat_file)
    nat_ckpt = torch.load(nat_root)
    nat_net.load_state_dict(nat_ckpt['net'])
    if args.nat_init:
        basic_net.load_state_dict(nat_ckpt['net'])

    criterion_ori = nn.CrossEntropyLoss()
    optimizer = optim.SGD(basic_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    if args.resume:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100], gamma = 1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], gamma = 0.1)

    classes_idx = i_class_idx(trainset.targets)
    if args.adversarial_test:
        test(start_epoch, testloader)
        adv_test(start_epoch)
    else:
        for epoch in range(start_epoch, args.num_epoches):
            classes_idx.get_idx()
            train(epoch)
            test(epoch, testloader)
