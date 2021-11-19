from torch.utils.data import Subset


class MutableSubset(Subset):
    """
    Mutable dataset subset, that will allow us to impute data as we train.
    """
    def __setitem__(self, key, value):
        self.dataset[key] = value
