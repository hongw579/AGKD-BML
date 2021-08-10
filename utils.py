import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from argument import parser, print_args, create_logger

def softmax_crossentropy_labelsmooth(pred, targets, lb_smooth=None):
    if lb_smooth:
        eps = lb_smooth
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, targets.unsqueeze(1), 1)
        one_hot = one_hot*(1-eps)+(1-one_hot)*eps/(n_class - 1)
        log_prb = F.log_softmax(pred, dim = 1)
        loss = -(one_hot*log_prb).sum(dim=1)
        loss = loss.mean()
    else:
        loss = F.cross_entropy(pred, targets)
    return loss

def CW_loss(logits, targets, margin = 50., reduce = False):
    n_class = logits.size(1)
    onehot_targets = one_hot_tensor(targets, n_class, targets.device)
    self_loss = torch.sum(onehot_targets * logits, dim=1)
    other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

    loss = -torch.sum(torch.clamp(self_loss - other_loss + margin, 0))

    if reduce:
        sample_num = onehot_targets.shape[0]
        loss = loss / sample_num

    return loss

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0), num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

def angular_distance(logits1, logits2):
    numerator = logits1.mul(logits2).sum(1)
    logits1_l2norm = logits1.mul(logits1).sum(1).sqrt()
    logits2_l2norm = logits2.mul(logits2).sum(1).sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)
    for i, _ in enumerate(numerator):
        if numerator[i]>denominator[i]:
            numerator[i]=denominator[i]
    D = torch.sub(1.0, torch.abs(torch.div(numerator, denominator)))
    return D

def L2_norm(logit):
    l2norm = logit.mul(logit) #.sum(1)
    return l2norm.mean() #.sqrt().mean()

def triplet_loss(a, p, n, margin, lam2):
    positive_dist = angular_distance(a, p)
    negative_dist = angular_distance(a, n)
    L_trip = margin+positive_dist-negative_dist
    L_trip = torch.max(torch.tensor(0., device = L_trip.device), L_trip).mean()
    norm = L2_norm(a) + L2_norm(p) + L2_norm(n)
    loss = L_trip + lam2*norm
    return loss

def MC_labels(logits, targets):
    a = logits.detach().clone()
    a[np.arange(targets.shape[0]),targets] = -1e4
    _, labels = a.max(1)
    return labels

def attention_map(feature):
    attention = torch.sum(feature, 1)
    attention_size = attention.shape
    attention_norm = attention.view(attention.size(0), -1)
    attention_norm -= attention_norm.min(1, keepdim=True)[0]
    attention_norm /= attention_norm.max(1, keepdim=True)[0]
    attention_norm = attention_norm.view(attention_size)
    return attention_norm

def mart_loss(logits, logits_adv, y, kl, beta=6.0):
    batch_size = len(logits)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_bce = F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_bce + float(beta) * loss_robust
    return loss

class i_class_idx(object):
    def __init__(self, targets):
        self.targets = targets
    def get_idx(self):
        self.i_idx = []
        for j in range(10):
            idx = np.where(np.array(self.targets) == j)[0]
            random.shuffle(idx)
            self.i_idx.append(idx)
    def get_batch(self, batch_targets, delete = False):
        x1_pos_idx = np.zeros(len(batch_targets), dtype = np.int)
        for j in range(len(batch_targets)):
            target = batch_targets[j].item()
            to_batch = np.random.choice(self.i_idx[target], 1, replace = False)
            x1_pos_idx[j] = to_batch
            if delete:
                self.i_idx[target] = np.setdiff1d(self.i_idx[target], to_batch, assume_unique = True)
        return x1_pos_idx

def to_variable(dataset, idx_list):
    for i, idx in enumerate(idx_list):
        if i == 0:
            input_batch = dataset[idx][0]
            input_batch.unsqueeze_(0)
            target_batch = torch.tensor(dataset[idx][1])
            target_batch.unsqueeze_(0)
        else:
            one_instance = dataset[idx][0]
            one_instance.unsqueeze_(0)
            input_batch = torch.cat([input_batch, one_instance], 0)
            one_target = torch.tensor(dataset[idx][1])
            one_target.unsqueeze_(0)
            target_batch = torch.cat([target_batch, one_target], 0)
    input_batch = Variable(input_batch)
    target_batch = Variable(target_batch)
    return input_batch, target_batch

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred = 0.1*torch.ones([5, 10], dtype=torch.float64,device = device)
    target = torch.tensor([1, 0, 1, 1, 1], device = device)
    loss = softmax_crossentropy_labelsmooth(pred, target, lb_smooth = 0.1)
    print(loss)
