import torch
import torch.nn as nn

class OODLoss(nn.Module):
    def __init__(self, ood_threshold):
        super(OODLoss, self).__init__()
        self.ood_threshold = ood_threshold

    def forward(self, in_dist_log_likelihood, ood_log_likelihood):
        # Compute the OOD label
        p_ood = 1 - torch.exp(in_dist_log_likelihood)

        # Compute the two terms in the loss
        ood_term = torch.log(p_ood + 1e-10)
        in_dist_term = in_dist_log_likelihood

        # Combine them into the loss
        loss = -0.5 * (ood_term.mean() + in_dist_term.mean())

        return loss
