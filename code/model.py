# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class BatchContrastiveLoss(nn.Module):
    def __init__(self, max_bs, device):
        super(BatchContrastiveLoss, self).__init__()
        self.loss_fct = CrossEntropyLoss()
        self.a_range = torch.arange(max_bs, device=device)

        
    def forward(self, code_vec, nl_vec, bs):
        scores = (nl_vec[:, None, :] * code_vec[None, :, :]).sum(-1)
        loss = self.loss_fct(scores, self.a_range[:bs])
        return loss


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, code_inputs, nl_inputs, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]

        if return_vec:
            return code_vec, nl_vec
        
        loss_fct = BatchContrastiveLoss(bs, code_inputs.device)
        return loss_fct(code_vec, nl_vec, bs), code_vec, nl_vec
