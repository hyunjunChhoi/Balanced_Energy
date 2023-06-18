import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.functional import normalize

def influence(value, weight, dim=None, keepdim=False):

    influence= torch.sum(value*F.relu(weight), dim=dim, keepdim=keepdim)
    influence = influence.squeeze(dim)

    return influence

def smooth(arr, lamda1):
    new_array = arr
    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]
    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2)) / 2
    return lamda1 * loss


def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def energy_loss_with_smooth_sparsity(logits, targets):
    ood_ind = 254
    void_ind = 255
    num_class = 19
    T = 1.
    m_in = -12
    m_out = -6

    energy = -(T * torch.logsumexp(logits[:, :num_class, :, :] / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


def balanced_energy_loss_with_smooth_sparsity(logits, targets, gamma1=0, gamma2=0, alpha=0):
    OOD_prior=torch.tensor([0.152509041,0.013298,0.106703,0.040058,0.033866,0.005256,0.001065,0.027989,0.080249,0.097971,0.002175,0.153123,0.029222,0.087057,0.119925,0.016023,0.007071,0.017048,0.009393]).cuda()
    
    OOD_prior_gamma1=OOD_prior**gamma1
    OOD_prior_gamma2=OOD_prior**gamma2

    OOD_prior_gamma1 = normalize(OOD_prior_gamma1, p=1.0, dim=0)
    OOD_prior_gamma2 = normalize(OOD_prior_gamma2, p=1.0, dim=0)

    OOD_prior_gamma1=OOD_prior_gamma1[None,:,None,None]
    OOD_prior_gamma2=OOD_prior_gamma2[None,:,None,None]

    ood_ind = 254
    void_ind = 255
    num_class = 19
    T = 1.0
    m_in = -12
    m_out = -6

    softmax=torch.nn.functional.softmax(logits[:, :num_class, :, :] ,dim=1)
    
    influences_for_margin =  influence(softmax,OOD_prior_gamma1,dim=1)    
    influences_for_loss =  influence(softmax,OOD_prior_gamma2,dim=1)

    Ec_out_margin=influences_for_margin[targets==ood_ind]
    Ec_out_weight=influences_for_loss[targets==ood_ind]

    energy = -(T * torch.logsumexp(logits[:, :num_class, :, :] / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]
    
    Ec_out= Ec_out-(alpha*Ec_out_margin)    

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        #loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + (torch.pow(F.relu((m_out - Ec_out)), 2) * Ec_out_weight).sum() / Ec_out_weight.sum())        
        
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


def energy_loss_pure(logits, targets):
    ood_ind = 254
    void_ind = 255
    num_class = 19
    T = 1.
    m_in = -12
    m_out = -6

    energy = -(T * torch.logsumexp(logits[:, :num_class, :, :] / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
    
    return loss, energy        
        
def balanced_energy_loss_pure(logits, targets, gamma1=0, gamma2=0, alpha=0):
    OOD_prior=torch.tensor([0.152509041,0.013298,0.106703,0.040058,0.033866,0.005256,0.001065,0.027989,0.080249,0.097971,0.002175,0.153123,0.029222,0.087057,0.119925,0.016023,0.007071,0.017048,0.009393]).cuda()
    
    OOD_prior_gamma1=OOD_prior**gamma1
    OOD_prior_gamma2=OOD_prior**gamma2

    OOD_prior_gamma1 = normalize(OOD_prior_gamma1, p=1.0, dim=0)
    OOD_prior_gamma2 = normalize(OOD_prior_gamma2, p=1.0, dim=0)

    OOD_prior_gamma1=OOD_prior_gamma1[None,:,None,None]
    OOD_prior_gamma2=OOD_prior_gamma2[None,:,None,None]

    ood_ind = 254
    void_ind = 255
    num_class = 19
    T = 1.0
    m_in = -12
    m_out = -6

    softmax=torch.nn.functional.softmax(logits[:, :num_class, :, :] ,dim=1)
    
    influences_for_margin =  influence(softmax,OOD_prior_gamma1,dim=1)    
    influences_for_loss =  influence(softmax,OOD_prior_gamma2,dim=1)

    Ec_out_margin=influences_for_margin[targets==ood_ind]
    Ec_out_weight=influences_for_loss[targets==ood_ind]

    energy = -( torch.logsumexp(logits[:, :num_class, :, :] / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]    
    
    Ec_out= Ec_out-(alpha*Ec_out_margin)

    loss = torch.tensor(0.).cuda()


    if Ec_out.size()[0] == 0:
        loss += (torch.pow(F.relu(Ec_in - m_in), 2)).mean()
    else:
        #loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu((m_out - Ec_out)), 2).mean())
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + (torch.pow(F.relu((m_out - Ec_out)), 2) * Ec_out_weight).sum() / Ec_out_weight.sum())

    return loss, energy        



class Gambler(torch.nn.Module):
    def __init__(self, reward, device, pretrain=-1, ood_reg=.1):
        super(Gambler, self).__init__()
        self.reward = torch.tensor([reward]).cuda(device)
        self.pretrain = pretrain
        self.ood_reg = ood_reg
        self.device = device

    def forward(self, pred, targets, wrong_sample=False):

        pred_prob = torch.softmax(pred, dim=1)

        assert torch.all(pred_prob > 0), print(pred_prob[pred_prob <= 0])
        assert torch.all(pred_prob <= 1), print(pred_prob[pred_prob > 1])
        true_pred, reservation = pred_prob[:, :-1, :, :], pred_prob[:, -1, :, :]

        # compute the reward via the energy score;
        reward = torch.logsumexp(pred[:, :-1, :, :], dim=1).pow(2)

        if reward.nelement() > 0:
            gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)
            reward = reward.unsqueeze(0)
            reward = gaussian_smoothing(reward)
            reward = reward.squeeze(0)
        else:
            reward = self.reward

        if wrong_sample:  # if there's ood pixels inside the image
            reservation = torch.div(reservation, reward)
            mask = targets == 254
            # mask out each of the ood output channel
            reserve_boosting_energy = torch.add(true_pred, reservation.unsqueeze(1))[mask.unsqueeze(1).
                repeat(1, 19, 1, 1)]
            
            gambler_loss_out = torch.tensor([.0], device=self.device)
            if reserve_boosting_energy.nelement() > 0:
                reserve_boosting_energy = torch.clamp(reserve_boosting_energy, min=1e-7).log()
                gambler_loss_out = self.ood_reg * reserve_boosting_energy

            # gambler loss for in-lier pixels
            void_mask = targets == 255
            targets[void_mask] = 0  # make void pixel to 0
            targets[mask] = 0  # make ood pixel to 0
            gambler_loss_in = torch.gather(true_pred, index=targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss_in = torch.add(gambler_loss_in, reservation)

            # exclude the ood pixel mask and void pixel mask
            gambler_loss_in = gambler_loss_in[(~mask) & (~void_mask)].log()
            return -(gambler_loss_in.mean() + gambler_loss_out.mean())
        else:
            mask = targets == 255
            targets[mask] = 0
            reservation = torch.div(reservation, reward)
            gambler_loss = torch.gather(true_pred, index=targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss = torch.add(gambler_loss, reservation)
            gambler_loss = gambler_loss[~mask].log()
            # assert not torch.any(torch.isnan(gambler_loss)), "nan check"
            return -gambler_loss.mean()
