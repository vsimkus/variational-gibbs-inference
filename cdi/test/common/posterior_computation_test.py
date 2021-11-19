import torch

from cdi.common.posterior_computation import exact_gibbs_inference, \
                                             exact_inference, \
                                             compute_xm_z_covariance


def test_exact_gibbs_inference_outputs():
    # Input data
    X = torch.tensor([[0, 1, -1, 20],
                      [11, 0, 0, 0],
                      [1, 2, 3, 4]],
                     dtype=torch.float)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    F = torch.tensor([[5, 3, 2],
                      [1, 2, 3],
                      [6, 7, 7],
                      [9, -1, 0]],
                     dtype=torch.float)
    cov = torch.tensor([3, 1, -1, 6], dtype=torch.float)
    mean = torch.tensor([0, -1, 2, 3], dtype=torch.float)

    # Method call
    _, _, means, covs = exact_gibbs_inference(X, M.type_as(X),
                                              F=F,
                                              cov=cov,
                                              mean=mean)

    means_populated = means != 0
    covs_populated = covs != 0
    vars_populated = torch.diagonal(covs_populated, dim1=-2, dim2=-1)

    assert torch.all(~(means_populated ^ ~M)),\
        'Means should only be returned for missing values.'

    assert covs_populated.sum() == vars_populated.sum(),\
        'Covariances should be diagonal.'

    assert torch.all(~(vars_populated ^ ~M)),\
        'Vars should only be returned for missing values.'


def test_exact_inference():
    # Input data
    X = torch.randn((10, 4), dtype=torch.double)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 0, 1],
                      [1, 1, 1, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1],
                      [0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]],
                     dtype=torch.double)
    F = torch.tensor([[5, 3, 2],
                      [1, 2, 3],
                      [6, 7, 7],
                      [9, -1, 0]],
                     dtype=torch.double)
    cov = torch.tensor([3, 1, 2, 6], dtype=torch.double)
    mean = torch.tensor([0, -1, 2, 3], dtype=torch.double)

    # Method call
    mean_z, cov_z, mean_x, cov_x = exact_inference(X, M,
                                                   F=F,
                                                   cov=cov,
                                                   mean=mean)
    mean_z = mean_z.squeeze()

    M = torch.diag_embed(M)
    # Alternative z posterior computation:
    latent_dim = F.shape[1]
    cov_z_true = torch.inverse(torch.eye(latent_dim)
                               + F.T @ M
                               @ torch.inverse(torch.diag(cov))
                               @ M @ F)
    assert torch.allclose(cov_z_true, cov_z),\
        'Posterior covariance matrix of the latents does not match!'

    mean_z_true = torch.empty((X.shape[0], latent_dim), dtype=torch.double)
    for i in range(X.shape[0]):
        mask = torch.diag(M[i]).bool()
        F_i = F[mask]
        cov_i = cov[mask]
        mean_i = mean[mask]
        x_i = X[i, mask]
        mean_z_true[i, :] = (cov_z_true[i, :, :] @ F_i.T
                             @ torch.inverse(torch.diag(cov_i))
                             @ (x_i - mean_i).T)

    assert torch.allclose(mean_z_true, mean_z),\
        'Posterior mean matrix of the latents does not match!'

    # Alternative x posterior computation
    mean_x_true = (F @ mean_z_true.T + mean.unsqueeze(-1)).T
    cov_x_true = F @ cov_z_true @ F.T + torch.diag(cov)
    assert torch.allclose(cov_x_true, cov_x),\
        'Posterior covariance matrix of the observables does not match!'
    assert torch.allclose(mean_x_true, mean_x),\
        'Posterior mean matrix of the observables does not match!'

    # Also, just for sanity - check posterior of x without setting missing
    # components to zero, but instead using truncated matrices, where missing
    # component rows/cols are removed
    for i in range(X.shape[0]):
        # Select observed dims
        mask = ~(torch.diag(M[i]).bool())
        F_i = F[mask]
        cov_i = cov[mask]
        mean_i = mean[mask]
        x_i = X[i, mask]
        mean_x_true2 = (F_i @ mean_z_true[i].T + mean_i).T
        cov_x_true2 = F_i @ cov_z_true[i] @ F_i.T + torch.diag(cov_i)

        cov_x_obs = (torch.masked_select(cov_x[i],
                                         mask * mask.unsqueeze(-1))
                     .reshape(mask.sum(), mask.sum()))
        assert torch.allclose(cov_x_true2, cov_x_obs),\
            'Posterior covariance matrix of the observables does not match!'
        assert torch.allclose(mean_x_true2, mean_x[i, mask]),\
            'Posterior mean matrix of the observables does not match!'


# Alternative, but similar test with different posterior computation
def test_exact_inference2():
    # Input data
    X = torch.randn((10, 4), dtype=torch.double)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 0, 1],
                      [1, 1, 1, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1],
                      [0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]],
                     dtype=torch.double)
    F = torch.tensor([[5, 3, 2],
                      [1, 2, 3],
                      [6, 7, 7],
                      [9, -1, 0]],
                     dtype=torch.double)
    cov = torch.tensor([3, 1, 2, 6], dtype=torch.double)
    mean = torch.tensor([0, -1, 2, 3], dtype=torch.double)

    # Method call
    mean_z, cov_z, mean_x, cov_x = exact_inference(X, M,
                                                   F=F,
                                                   cov=cov,
                                                   mean=mean)
    mean_z = mean_z.squeeze()

    M = torch.diag_embed(M)
    # Alternative z posterior computation:
    latent_dim = F.shape[1]
    # cov_z_true = torch.inverse(torch.eye(latent_dim)
    #                            + F.T @ M
    #                            @ torch.inverse(torch.diag(cov))
    #                            @ M @ F)
    cov_z_true = torch.empty((X.shape[0], latent_dim, latent_dim), dtype=torch.double)
    for i in range(X.shape[0]):
        mask = torch.diag(M[i]).bool()
        F_i = F[mask]
        cov_i = cov[mask]
        mean_i = mean[mask]
        x_i = X[i, mask]
        cov_z_true[i, :] = torch.inverse(torch.eye(latent_dim) + F_i.T @ torch.diag(1/cov_i) @ F_i)
        # cov_z_true2 = torch.eye(latent_dim) - F_i.T @ torch.inverse(F_i @ F_i.T + torch.diag(cov_i)) @ F_i
        # assert torch.allclose(cov_z_true[i, :], cov_z_true2)

    assert torch.allclose(cov_z_true, cov_z),\
        'Posterior covariance matrix of the latents does not match!'

    mean_z_true = torch.empty((X.shape[0], latent_dim), dtype=torch.double)
    for i in range(X.shape[0]):
        mask = torch.diag(M[i]).bool()
        F_i = F[mask]
        cov_i = cov[mask]
        mean_i = mean[mask]
        x_i = X[i, mask]
        mean_z_true[i, :] = (cov_z_true[i, :, :] @ F_i.T
                             @ torch.diag(1/cov_i)
                             @ (x_i - mean_i).T)

    assert torch.allclose(mean_z_true, mean_z),\
        'Posterior mean matrix of the latents does not match!'

    # Alternative posterior computation
    for i in range(X.shape[0]):
        # Select observed dims
        mask = torch.diag(M[i]).bool()
        F_o = F[mask]
        cov_o = cov[mask]
        mean_o = mean[mask]
        F_m = F[~mask]
        cov_m = cov[~mask]
        mean_m = mean[~mask]
        x_o = X[i, mask]

        mean_z_true2 = F_o.T @ torch.inverse(F_o @ F_o.T + torch.diag(cov_o)) @ (x_o - mean_o)
        cov_z_true2 = torch.eye(latent_dim) - F_o.T @ torch.inverse(F_o @ F_o.T + torch.diag(cov_o)) @ F_o
        mean_x_true2 = mean_m + F_m @ mean_z_true2
        cov_x_true2 = torch.diag(cov_m) + F_m @ cov_z_true2 @ F_m.T

        assert torch.allclose(cov_z_true2, cov_z[i]),\
            'Posterior covariance matrix of the latents does not match!'
        assert torch.allclose(mean_z_true2, mean_z[i]),\
            'Posterior mean matrix of the latents does not match!'

        cov_x_mis = (torch.masked_select(cov_x[i],
                                         ~mask * (~mask).unsqueeze(-1))
                     .reshape((~mask).sum(), (~mask).sum()))
        assert torch.allclose(cov_x_true2, cov_x_mis),\
            'Posterior covariance matrix of the missing does not match!'
        assert torch.allclose(mean_x_true2, mean_x[i, ~mask]),\
            'Posterior mean matrix of the missing does not match!'


def test_compute_xm_z_covariance():
    # Input data
    X = torch.randn((10, 4), dtype=torch.double)
    M = torch.tensor([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 0, 1],
                      [1, 1, 1, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1],
                      [0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]],
                     dtype=torch.bool)
    F = torch.tensor([[5, 3, 2],
                      [1, 2, 3],
                      [6, 7, 7],
                      [9, -1, 0]],
                     dtype=torch.double)
    cov = torch.tensor([3, 1, 2, 6], dtype=torch.double)
    mean = torch.tensor([0, -1, 2, 3], dtype=torch.double)

    # Method call
    mean_z, cov_z, mean_x, cov_x = exact_inference(X, M,
                                                   F=F,
                                                   cov=cov,
                                                   mean=mean)
    cov_xm_z = compute_xm_z_covariance(M, F, cov_z)

    # Alternative posterior computation
    for i in range(X.shape[0]):
        # Select observed dims
        mask = M[i]
        F_o = F[mask]
        cov_o = cov[mask]
        F_m = F[~mask]

        cov_xm_z_true = F_m - F_m @ F_o.T @ torch.inverse(F_o @ F_o.T + torch.diag(cov_o)) @ F_o

        assert torch.allclose(cov_xm_z_true, cov_xm_z[i][~mask]),\
            'Posterior conditional covariance matrix between xm and z does not match!'
