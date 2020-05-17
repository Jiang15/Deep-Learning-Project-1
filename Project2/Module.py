import torch


class Module(object):
    def _init_(self):
        self.module = 0

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output):
        """Copied from autograd: https://github.com/pytorch/pytorch/blob/master/torch/autograd/function.py"""
        r"""Defines a formula for differentiating the operation.
        This function is to be overridden by all subclasses.
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs did :func:`forward` return, and it should return as many
        tensors, as there were inputs to :func:`forward`. Each argument is the
        gradient w.r.t the given output, and each returned value should be the
        gradient w.r.t. the corresponding input.
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computated w.r.t. the
        output.
        """
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass
        # from autograd
        # for p in self.param():
        #     if p.grad is not None:
        #         p.grad.detach_()
        #         p.grad.zero_()

    def reset(self):
        pass

class Parameters(Module):
    def __init__(self,value):
        super(Parameters,self).__init__()
        self.value = value
        self.grad = torch.zeros_like(self.value)
