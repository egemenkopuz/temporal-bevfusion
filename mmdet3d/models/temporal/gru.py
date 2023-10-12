import torch
from torch import nn
from torch.autograd import Variable

from mmdet3d.models.builder import TEMPORAL


@TEMPORAL.register_module()
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(
            self.input_size + self.hidden_size,
            2 * self.hidden_size,
            3,
            padding=self.kernel_size // 2,
        )
        self.Conv_ct = nn.Conv2d(
            self.input_size + self.hidden_size, self.hidden_size, 3, padding=self.kernel_size // 2
        )

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            device = input.get_device()
            if device == -1:
                hidden = Variable(torch.zeros(size_h))
            else:
                hidden = Variable(torch.zeros(size_h)).to(device)
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = torch.sigmoid(rt)
        update_gate = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = torch.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h
