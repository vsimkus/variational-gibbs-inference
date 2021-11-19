import unittest.mock as mock

import jsonargparse
import torch
import torch.testing as tt
from pytest import fixture

from cdi.models.variational_distribution import (GaussianVarDistr,
                                                 GaussianVarDistrFast,
                                                 SharedGaussianVarDistr3)
from cdi.util.arg_utils import convert_namespace
from cdi.util.test_utils import BooleanTensorMatcher, TensorMatcher


@fixture
def gaus_q():
    args = ['--var_model.input_dim', '3',
            '--var_model.hidden_dims', '10', '4',
            '--var_model.activation', 'lrelu',
            '--var_model.input_missing_vectors', 'False',
            '--var_model.num_univariate_q', '4']

    argparser = jsonargparse.ArgumentParser()
    argparser = GaussianVarDistr.add_model_args(argparser)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    return GaussianVarDistr(args)


def test_gaus_q_outputs(gaus_q):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)

    # Method call
    means, log_vars = gaus_q(X, M.type_as(X),
                             M_selected=M,
                             sample_all=True)

    means_populated = means != 0
    log_vars_populated = log_vars != float('-inf')

    assert torch.all(~(means_populated ^ ~M)),\
        'Means should only be returned for missing values.'

    assert torch.all(~(log_vars_populated ^ ~M)),\
        'Log-vars should only be returned for missing values.'


@fixture
def shared_gaus_q3():
    args = ['--var_model.input_dim', '4',
            '--var_model.prenet_layer_dims', '10',
            '--var_model.outnet_layer_dims', '4',
            '--var_model.activation', 'lrelu',
            '--var_model.input_missing_vectors', 'missing',
            '--var_model.set_missing_zero', 'True']

    argparser = jsonargparse.ArgumentParser()
    argparser = SharedGaussianVarDistr3.add_model_args(argparser)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    return SharedGaussianVarDistr3(args)


def test_shared_gaus_q3_outputs_sample_all(shared_gaus_q3):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)

    # Method call
    means, log_vars = shared_gaus_q3(X, M.type_as(X),
                                     M_selected=M,
                                     sample_all=True)

    means_populated = means != 0
    log_vars_populated = log_vars != float('-inf')

    assert torch.all(~(means_populated ^ ~M)),\
        'Means should only be returned for missing values.'

    assert torch.all(~(log_vars_populated ^ ~M)),\
        'Log-vars should only be returned for missing values.'


def test_shared_gaus_q3_outputs_sample_one(shared_gaus_q3):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    M_selected = torch.tensor([[0, 1, 1, 1],
                               [1, 1, 0, 1],
                               [1, 1, 1, 1]],
                              dtype=torch.bool)

    # Method call
    means, log_vars = shared_gaus_q3(X, M.type_as(X),
                                     M_selected=M_selected,
                                     sample_all=False)

    means_populated = means != 0
    log_vars_populated = log_vars != float('-inf')

    assert torch.all(~(means_populated ^ ~M_selected)),\
        'Means should only be returned for missing values.'

    assert torch.all(~(log_vars_populated ^ ~M_selected)),\
        'Log-vars should only be returned for missing values.'


def test_shared_gaus_q3_model_call(shared_gaus_q3):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)

    # Mock outputs
    mean = torch.tensor([[1, 0, 0, 0],
                         [0, 2, 0, 0],
                         [0, 0, 4, 0],
                         [0, 0, 0, 9]],
                        dtype=torch.float)
    log_var = torch.tensor([[6, 0, 0, 0],
                            [0, 7, 0, 0],
                            [0, 0, 8, 0],
                            [0, 0, 0, 11]],
                           dtype=torch.float)

    # True call inputs
    X_true = torch.tensor([[0, 1, -1, 20],
                           [11, 0, 0, 0],
                           [11, 0, 0, 0],
                           [11, 0, 0, 0]],
                          dtype=torch.float)
    M_true = torch.tensor([[0, 1, 1, 1],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0]],
                          dtype=torch.bool)
    M_selected_true = torch.tensor([[0, 1, 1, 1],
                                    [1, 0, 1, 1],
                                    [1, 1, 0, 1],
                                    [1, 1, 1, 0]],
                                   dtype=torch.bool)

    # True outputs
    means_true = torch.tensor([[1, 0, 0, 0],
                               [0, 2, 4, 9],
                               [0, 0, 0, 0]],
                              dtype=torch.float)
    log_vars_true = torch.tensor([[6, float('-inf'), float('-inf'), float('-inf')],
                                  [float('-inf'), 7, 8, 11],
                                  [float('-inf'), float('-inf'), float('-inf'), float('-inf')]],
                                 dtype=torch.float)

    # Mocks
    shared_gaus_q3.model.forward = mock.MagicMock(
        return_value=(mean, log_var)
    )

    # Method call
    means, log_vars = shared_gaus_q3(X, M.type_as(X),
                                     M_selected=M,
                                     sample_all=True)

    shared_gaus_q3.model.forward.assert_called_once_with(
        TensorMatcher(X_true), TensorMatcher(M_true.type_as(X)),
        BooleanTensorMatcher(M_selected_true)
    )

    means_populated = means != 0
    log_vars_populated = log_vars != float('-inf')

    assert torch.all(~(means_populated ^ ~M)),\
        'Means should only be returned for missing values.'

    assert torch.all(~(log_vars_populated ^ ~M)),\
        'Log-vars should only be returned for missing values.'

    tt.assert_allclose(means, means_true)
    tt.assert_allclose(log_vars, log_vars_true)


@fixture
def gaus_q_fast():
    args = ['--var_model.input_dim', '4',
            '--var_model.hidden_dims', '10', '5',
            '--var_model.activation', 'lrelu',
            '--var_model.input_missing_vectors', 'False']

    argparser = jsonargparse.ArgumentParser()
    argparser = GaussianVarDistrFast.add_model_args(argparser)
    args = argparser.parse_args(args)

    args = convert_namespace(args)

    return GaussianVarDistrFast(args)


def test_gauss_q_fast_doesnt_leak(gaus_q_fast):
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 3, 4],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)

    means1, log_vars1 = gaus_q_fast(X, M.type_as(X), M_selected=M)
    means1_cp = means1.clone()
    log_vars1_cp = log_vars1.clone()

    # Wiggle an input value and check if output changes
    X2 = X.clone()
    X2[0, 2] = 39.

    means2, log_vars2 = gaus_q_fast(X2, M.type_as(X2), M_selected=M)

    which_change = torch.where(torch.arange(X.shape[-1]) != 2)[0]
    means1[0, which_change] = 0.
    means2[0, which_change] = 0.
    log_vars1[0, which_change] = 0.
    log_vars2[0, which_change] = 0.

    tt.assert_allclose(means1, means2)
    tt.assert_allclose(log_vars1, log_vars2)

    # Try wiggling more input values
    X3 = X.clone()
    X3[0, 3] = 66.
    X3[2, 1] = -100.

    means3, log_vars3 = gaus_q_fast(X3, M.type_as(X3), M_selected=M)

    means1 = means1_cp.clone()
    log_vars1 = log_vars1_cp.clone()
    which_change0 = torch.where(torch.arange(X.shape[-1]) != 3)[0]
    which_change2 = torch.where(torch.arange(X.shape[-1]) != 1)[0]
    means1[0, which_change0] = 0.
    means1[2, which_change2] = 0.
    means3[0, which_change0] = 0.
    means3[2, which_change2] = 0.
    log_vars1[0, which_change0] = 0.
    log_vars1[2, which_change2] = 0.
    log_vars3[0, which_change0] = 0.
    log_vars3[2, which_change2] = 0.

    tt.assert_allclose(means1, means3)
    tt.assert_allclose(log_vars1, log_vars3)
