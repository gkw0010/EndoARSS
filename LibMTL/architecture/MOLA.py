import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture

class MOLA(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MOLA, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self.encoder = self.encoder_class()
        
    def forward(self, inputs, task_name=None):
        s_rep = self.encoder(inputs)
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            ss_rep = self._prepare_rep(s_rep, task, same_rep=False)
            out[task] = self.decoders[task](ss_rep)
        return out
    
