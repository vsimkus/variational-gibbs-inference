import numpy as np
import torch

from cdi.util.data.data_augmentation_dataset import DataAugmentation


def test_data_augmentation_dataset():
    # Original data
    X = torch.tensor([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5],
                      [1, 0, 4, 0, 6],
                      [1, 1, 1, 5, 5],
                      [9, 0, 8, 2, 7]], dtype=torch.float)
    M = torch.tensor([[1, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1],
                      [1, 1, 1, 1, 0],
                      [1, 0, 1, 1, 1]], dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    # True outputs
    X_true = torch.tensor([[1, 2, 3, 4, 5],
                           [1, 2, 3, 4, 5],
                           [1, 2, 3, 4, 5],
                           [9, 8, 7, 6, 5],
                           [1, 0, 4, 0, 6],
                           [1, 0, 4, 0, 6],
                           [1, 0, 4, 0, 6],
                           [1, 1, 1, 5, 5],
                           [1, 1, 1, 5, 5],
                           [1, 1, 1, 5, 5],
                           [9, 0, 8, 2, 7],
                           [9, 0, 8, 2, 7],
                           [9, 0, 8, 2, 7]], dtype=torch.float)
    M_true = torch.tensor([[1, 0, 0, 1, 1],
                           [1, 0, 0, 1, 1],
                           [1, 0, 0, 1, 1],
                           [1, 1, 1, 1, 1],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 1],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 0],
                           [1, 0, 1, 1, 1],
                           [1, 0, 1, 1, 1],
                           [1, 0, 1, 1, 1]], dtype=torch.bool)

    I_true = torch.tensor([0, 5, 9, 1, 2, 6, 10, 3, 7, 11, 4, 8, 12],
                          dtype=torch.long)
    OI_true = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                           dtype=torch.long)
    incomp_mask_true = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    dtype=torch.bool)

    # Method call
    augmented = DataAugmentation(torch.utils.data.TensorDataset(X, M, I),
                                 num_copies=3)
    X_out, M_out, I_out, OI_out, incomp_mask_out = augmented[:]

    assert np.allclose(X_out, X_true),\
        'The augmented data is not correct!'

    assert np.allclose(M_out, M_true),\
        'The augmented data is not correct!'

    assert np.allclose(I_out, I_true),\
        'Indices are not correct!'

    assert np.allclose(OI_out, OI_true),\
        'Original indices are not correct!'

    assert np.allclose(incomp_mask_out, incomp_mask_true),\
        'Incomplete mask is incorrect!'

    # Select slice
    X_out, M_out, I_out, OI_out, incomp_mask_out = augmented[:3]

    assert np.allclose(X_out, X_true[:7]),\
        'The augmented data is not correct!'

    assert np.allclose(M_out, M_true[:7]),\
        'The augmented data is not correct!'

    assert np.allclose(I_out, I_true[:7]),\
        'Indices are not correct!'

    assert np.allclose(OI_out, OI_true[:7]),\
        'Original indices are not correct!'

    assert np.allclose(incomp_mask_out, incomp_mask_true[:7]),\
        'Incomplete mask is incorrect!'

    # Select slice
    X_out, M_out, I_out, OI_out, incomp_mask_out = augmented[-3:]

    assert np.allclose(X_out, X_true[-9:]),\
        'The augmented data is not correct!'

    assert np.allclose(M_out, M_true[-9:]),\
        'The augmented data is not correct!'

    assert np.allclose(I_out, I_true[-9:]),\
        'Indices are not correct!'

    assert np.allclose(OI_out, OI_true[-9:]),\
        'Original indices are not correct!'

    assert np.allclose(incomp_mask_out, incomp_mask_true[-9:]),\
        'Incomplete mask is incorrect!'

    # Select one complete sample
    X_out, M_out, I_out, OI_out, incomp_mask_out = augmented[1]

    assert np.allclose(X_out, X_true[3]),\
        'The augmented data is not correct!'

    assert np.allclose(M_out, M_true[3]),\
        'The augmented data is not correct!'

    assert np.allclose(I_out, I_true[3]),\
        'Indices are not correct!'

    assert np.allclose(OI_out, OI_true[3]),\
        'Original indices are not correct!'

    assert np.allclose(incomp_mask_out, incomp_mask_true[3]),\
        'Incomplete mask is incorrect!'

    # Select one incomplete sample
    X_out, M_out, I_out, OI_out, incomp_mask_out = augmented[0]

    assert np.allclose(X_out, X_true[0:3]),\
        'The augmented data is not correct!'

    assert np.allclose(M_out, M_true[0:3]),\
        'The augmented data is not correct!'

    assert np.allclose(I_out, I_true[0:3]),\
        'Indices are not correct!'

    assert np.allclose(OI_out, OI_true[0:3]),\
        'Original indices are not correct!'

    assert np.allclose(incomp_mask_out, incomp_mask_true[0:3]),\
        'Incomplete mask is incorrect!'


def test_data_augmentation_dataset2():
    # Original data
    X = torch.tensor([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5],
                      [1, 0, 4, 0, 6],
                      [1, 1, 1, 5, 5],
                      [9, 0, 8, 2, 7]], dtype=torch.float)
    M = torch.tensor([[1, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 1, 1, 1]], dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    # True outputs
    X_true = torch.tensor([[1, 2, 3, 4, 5],
                           [1, 2, 3, 4, 5],
                           [1, 2, 3, 4, 5],
                           [9, 8, 7, 6, 5],
                           [1, 0, 4, 0, 6],
                           [1, 0, 4, 0, 6],
                           [1, 0, 4, 0, 6],
                           [1, 1, 1, 5, 5],
                           [9, 0, 8, 2, 7],
                           [9, 0, 8, 2, 7],
                           [9, 0, 8, 2, 7]], dtype=torch.float)
    M_true = torch.tensor([[1, 0, 0, 1, 1],
                           [1, 0, 0, 1, 1],
                           [1, 0, 0, 1, 1],
                           [1, 1, 1, 1, 1],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 0, 1, 1, 1],
                           [1, 0, 1, 1, 1],
                           [1, 0, 1, 1, 1]], dtype=torch.bool)

    I_true = torch.tensor([0, 5, 8, 1, 2, 6, 9, 3, 4, 7, 10],
                          dtype=torch.long)
    OI_true = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4], dtype=torch.long)
    incomp_mask_true = torch.tensor([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                                    dtype=torch.bool)

    # Method call
    augmented = DataAugmentation(torch.utils.data.TensorDataset(X, M, I),
                                 num_copies=3)
    X_out, M_out, I_out, OI_out, incomp_mask_out = augmented[:]

    assert np.allclose(X_out, X_true),\
        'The augmented data is not correct!'

    assert np.allclose(M_out, M_true),\
        'The augmented data is not correct!'

    assert np.allclose(I_out, I_true),\
        'Indices are not correct!'

    assert np.allclose(OI_out, OI_true),\
        'Original indices are not correct!'

    assert np.allclose(incomp_mask_out, incomp_mask_true),\
        'Imcomplete mask is incorrect!'


def test_data_augmentation_dataset3():
    # Original data
    X = torch.tensor([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5],
                      [1, 0, 4, 0, 6],
                      [1, 1, 1, 5, 5],
                      [9, 0, 8, 2, 7]], dtype=torch.float)
    M = torch.tensor([[1, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 1, 1, 1]], dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    # True data
    X_true = torch.tensor([[1, 2, 3, 4, 5],
                           [9, 8, 7, 6, 5],
                           [1, 0, 4, 0, 6],
                           [1, 1, 1, 5, 5],
                           [9, 0, 8, 2, 7]], dtype=torch.float)
    M_true = torch.tensor([[1, 0, 0, 1, 1],
                           [1, 1, 1, 1, 1],
                           [0, 0, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 0, 1, 1, 1]], dtype=torch.bool)
    I_true = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    OI_true = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    incomp_mask_true = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)

    # Method call
    augmented = DataAugmentation(torch.utils.data.TensorDataset(X, M, I),
                                 num_copies=1)
    X_out, M_out, I_out, OI_out, incomp_mask_out = augmented[:]

    assert np.allclose(X_out, X_true),\
        'The augmented data should be same as original!'
    assert np.allclose(M_out, M_true),\
        'The augmented data should be same as original!'
    assert np.allclose(I_out, I_true),\
        'Indices are not correct!'

    assert np.allclose(OI_out, OI_true),\
        'Original indices are not correct!'

    assert np.allclose(incomp_mask_out, incomp_mask_true),\
        'Imcomplete mask is incorrect!'


def test_data_augmentation_dataset_set_value():
    # Original data
    X = torch.tensor([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5],
                      [1, 0, 4, 0, 6],
                      [1, 1, 1, 5, 5],
                      [9, 0, 8, 2, 7]], dtype=torch.float)
    M = torch.tensor([[1, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1],
                      [1, 1, 1, 1, 0],
                      [1, 0, 1, 1, 1]], dtype=torch.bool)
    I = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    # Method call
    augmented = DataAugmentation(torch.utils.data.TensorDataset(X, M, I),
                                 num_copies=3)
    X_out, M_out, I_out = augmented[:][:3]
    print('I_out', I_out)

    # Set whole dataset
    X_out = torch.randn_like(torch.tensor(X_out))

    augmented[I_out] = X_out
    X_out2, M_out2, I_out2 = augmented[:][:3]
    print('I_out2', I_out2)

    assert np.allclose(X_out, X_out2),\
        'The changed data is not persisted!'

    # Set a few values
    X_out2[0, 1] = -3
    X_out2[11, 2] = -4
    X_out2[12, 3] = -5

    augmented[I_out2[[0, 11, 12]]] = X_out2[[0, 11, 12], :]
    X_out3, M_out3, I_out3 = augmented[:][:3]

    assert np.allclose(X_out2, X_out3),\
        'The changed data is not persisted!'

    # Set a few values directly
    augmented[I_out2[[1, 3, 12]], [1, 3, 4]] = [5, 6, 7]
    X_out4, M_out4, I_out4 = augmented[:][:3]

    assert np.allclose(X_out4[[1, 3, 12], [1, 3, 4]], [5, 6, 7]),\
        'The changed data is not persisted!'
