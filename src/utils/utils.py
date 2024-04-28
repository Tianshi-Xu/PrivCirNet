import torch
import torch.nn.functional as F
class KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target, T=1.0):
        output = output[0] if isinstance(output, tuple) else output
        target = target[0] if isinstance(target, tuple) else target
        output, target = output / T, target / T
        target_prob = F.softmax(target, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        loss = - torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss