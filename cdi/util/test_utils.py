import torch


class TensorMatcher:
    def __init__(self, some_obj):
        self.some_obj = some_obj

    def __eq__(self, other):
        return torch.all(torch.eq(self.some_obj, other))


class BooleanTensorMatcher:
    def __init__(self, some_obj):
        self.some_obj = some_obj

    def __eq__(self, other):
        return torch.all(~(other ^ self.some_obj))
