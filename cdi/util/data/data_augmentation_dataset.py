import numpy as np
import torch
from torch.utils.data import Dataset


# Collate function to concat copies of sample into batch
def collate_augmented_samples(batch):
    return tuple((torch.cat([torch.as_tensor(sample)
                             for sample in samples])
                  for samples in zip(*batch)))


class DataAugmentation(Dataset):
    """
    Wrapper that augments incomplete samples.
    Maintains the order of input dataset, but the incomplete samples
    are repeated num_copies times.
    Overrides the Index matrix with its own.
    """
    def __init__(self, dataset, num_copies, augment_complete=False):
        self.dataset = dataset
        self.unaugmented_length = len(dataset)
        self.augment_complete = augment_complete

        # Get the unaugmented data
        # We ignore the original index, because
        # we will be making a complete copy of the
        # original data, with its own index
        X, M, I = dataset[:][:3]
        self.sample_incomp_mask = M.sum(axis=1) != X.shape[-1]
        if isinstance(self.sample_incomp_mask, torch.Tensor):
            self.sample_incomp_mask = self.sample_incomp_mask.numpy()

        if not self.augment_complete:
            # Set the observed data
            self.aug_data = X
            self.missing_mask = M
            self.original_idx = I

            # Create augmented copies
            X_augmented = np.tile(X[self.sample_incomp_mask, :], (num_copies-1, 1))
            M_augmented = np.tile(M[self.sample_incomp_mask, :], (num_copies-1, 1))
            I_augmented = np.tile(I[self.sample_incomp_mask], (num_copies-1))
            incomp_mask_augmented = np.ones_like(I_augmented, dtype=np.bool)
            self.aug_data = np.append(self.aug_data, X_augmented, axis=0)
            self.missing_mask = np.append(self.missing_mask, M_augmented, axis=0)
            self.original_idx = np.append(self.original_idx, I_augmented, axis=0)

            # Augmentation reference index
            # So that we can lookup all copies of a datapoint
            # E.g. for a dataset
            #        X                 M
            # [[0, 1, 2, 3],    [[1, 1, 1, 0],
            #  [1, 1, 1, 5],     [0, 0, 1, 1],
            #  [6, 9, 1, 2]]     [1, 1, 1, 1]]
            #
            #    aug_data
            # [[0, 1, 2, 3],
            #  [1, 1, 1, 5],
            #  [6, 9, 1, 2],
            #  [0, 1, 2, 3],
            #  [1, 1, 1, 5]
            #  [0, 1, 2, 3],
            #  [1, 1, 1, 5]]
            #
            # create aug_idx:
            # [[0, 3, 5], # The copies for 0th sample are in indices 0, 3, 5
            #  [1, 4, 6], # The copies for 1st sample are in indices 1, 4, 6
            #  [2, X, X]] # The 2nd sample doesn't have copies, where
            # X will be larger than the dataset, so that using it would cause
            # an error.
            self.placeholder_val = self.aug_data.shape[0] + 1
            self.aug_idx = np.full((X.shape[0], num_copies),
                                # Set initial value to larger than the dataset
                                self.placeholder_val,
                                dtype=np.long)
            self.aug_idx[:, 0] = np.arange(X.shape[0])

            if num_copies > 1:
                self.aug_idx[self.sample_incomp_mask, 1:] = \
                    np.arange(X.shape[0],
                            X.shape[0] + self.sample_incomp_mask.sum()*(num_copies-1))\
                    .reshape(-1, num_copies-1, order='F')

            # Cache a mask, which shows which values in the aug_idx are _not_
            # placeholders. I.e. shows that the element is either true value,
            # or an augmentation, but not a placeholder index.
            self.value_mask = self.aug_idx != self.placeholder_val

            # Store a cache mask indicating which samples are incomplete
            self.sample_incomp_mask = np.append(self.sample_incomp_mask,
                                                incomp_mask_augmented, axis=0)

        else:
            self.aug_data = np.tile(X, (num_copies, 1))
            self.missing_mask = np.tile(M, (num_copies, 1))
            self.original_idx = np.tile(I, (num_copies))
            self.sample_incomp_mask = np.tile(self.sample_incomp_mask, (num_copies))

            self.aug_idx = np.arange(0, self.aug_data.shape[0]).reshape(-1, num_copies, order='F')
            self.value_mask = np.ones_like(self.aug_idx, dtype=np.bool)

    def __getitem__(self, idx):
        # Augmented indices of this sample
        if self.augment_complete:
            indices = self.aug_idx[idx].flatten()
        else:
            mask = self.value_mask[idx]
            indices = self.aug_idx[idx][mask].flatten()

        return (self.aug_data[indices],
                self.missing_mask[indices],
                indices,
                self.original_idx[indices],
                self.sample_incomp_mask[indices])

    def get_all_data(self):
        # Augmented indices of this sample
        mask = self.value_mask[:]
        indices = self.aug_idx[:][mask].flatten()

        return (self.aug_data[indices],
                self.missing_mask[indices],
                indices,
                self.original_idx[indices],
                self.sample_incomp_mask[indices])

    def __setitem__(self, key, value):
        self.aug_data[key] = value

    def __len__(self):
        return self.unaugmented_length

    def augmented_len(self):
        return len(self.aug_data)

    def unaugmented_data(self):
        indices = self.aug_idx[:, 0]
        return (self.aug_data[indices],
                self.missing_mask[indices],
                indices,
                self.original_idx[indices],
                self.sample_incomp_mask[indices])


class DataAugmentationWithScheduler(DataAugmentation):
    def __init__(self, dataset, num_copies_scheduler, augment_complete=False):
        self.num_copies_scheduler = num_copies_scheduler
        num_copies = num_copies_scheduler.get_max_value()

        # Add specific assertion to make sure we don't reduce
        # the number of chains, why would we?
        scheduler_values = num_copies_scheduler.values
        assert (all(scheduler_values[i] <= scheduler_values[i + 1]
                    for i in range(len(scheduler_values)-1))),\
            'Number of chains should not be decreasing!'

        assert num_copies == self.num_copies_scheduler.get_last_value(), \
            'The maximum number of copies should be the same as last.'

        super().__init__(dataset, num_copies, augment_complete)

    def __getitem__(self, idx):
        # Get number of copies at this epoch
        num_copies = self.num_copies_scheduler.get_value()

        if self.augment_complete:
            indices = self.aug_idx[idx, :num_copies].flatten()
        else:
            # Augmented indices of this sample
            mask = self.value_mask[idx, :num_copies]
            indices = self.aug_idx[idx, :num_copies][mask].flatten()

        return (self.aug_data[indices],
                self.missing_mask[indices],
                indices,
                self.original_idx[indices],
                self.sample_incomp_mask[indices])

    def which_samples_are_new(self, prev_num_copies, new_num_copies, indices):
        all_new_indices = self.aug_idx[:, prev_num_copies:new_num_copies].flatten()

        return np.in1d(indices, all_new_indices)
