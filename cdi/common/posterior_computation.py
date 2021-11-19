import torch
import numpy as np


def exact_gibbs_inference(X, M, F, cov, mean):
    exact_means = torch.zeros_like(X)
    exact_vars = torch.zeros_like(X)
    M_not = 1. - M
    for d in range(M.shape[-1]):
        M_clone = M.clone()
        feature_indices = torch.arange(M_clone.shape[-1])
        feature_indices = feature_indices[feature_indices != d]
        indices = M_not[:, d].nonzero(as_tuple=False).flatten()

        # Prepare X tensor and missingness mask,
        # where each row can have a maximum of one missing value
        M_clone = M_clone[indices, :]
        M_clone[:, feature_indices] = 1.
        X_temp = X[indices, :]

        # Compute using exact posterior
        _, _, exact_mean, exact_cov = exact_inference(X_temp, M_clone,
                                                      F=F,
                                                      cov=cov,
                                                      mean=mean)

        M_clone_not = 1. - M_clone
        exact_mean *= M_clone_not
        exact_var = torch.diagonal(exact_cov, dim1=-2, dim2=-1) * M_clone_not

        exact_means[indices, d] = exact_mean[:, d]
        exact_vars[indices, d] = exact_var[:, d]

    covs = torch.diag_embed(exact_vars)
    return None, None, exact_means.detach(), covs.detach()


# Implementation of exact inference of the posterior distribution in
# Factor Analysis model, from Williams et. al (2018). Autoencoders and
# Probabilistic Inference with Missing Data: An Exact Solution for The
# Factor Analysis Case.
# ArXiv Preprint, 1801.03851. Retrieved from http://arxiv.org/abs/1801.03851

def exact_inference(X, M, F, cov, mean):
    """
    Computes the posterior distribution parameters of missing X,
    given the observed X.
    Args:
        X (N, D): input image batch
        M (N, D): image binary mask, 0 - missing, 1 - observed.
        F: factor loadings
        cov (D): prior diagonal covariance
        mean (D): prior mean
    Returns:
        mu_z: posterior mean of the latents, given the observed values in X
        sigma_z: posterior covariance of the latents, given the observed
            values in X
        mu_x: posterior mean of the observable values, given the observed
            values in X used to find the mean of the missing values
        sigma_x: posterior covariance of the observable values, given the
            observed values in X used to find the distribution of the
            missing values
    """
    # Compute common
    latent_dim = F.shape[1]
    M_inv_cov = M * (1/cov)

    # Calc posterior latent params
    sigma_z = torch.inverse(torch.eye(latent_dim,
                                      device=X.device,
                                      dtype=X.dtype)
                            + torch.matmul(F.t(),
                                           M_inv_cov.unsqueeze(-1) * F)
                            )
    mu_z = torch.matmul(sigma_z,
                        torch.matmul((M_inv_cov * (X - mean)), F).unsqueeze(-1)
                        )

    # Calc observation posterior
    mu_x, sigma_x = observable_posterior(mu_z, sigma_z, F, cov, mean)

    return mu_z, sigma_z, mu_x, sigma_x


def compute_xm_z_covariance(M, F, sigma_z):
    """
    Computes the covariance between x_m and z given x_o.
    Uses sigma_z (conditional posterior covariance on x_o), NOTE that
    it does *not depend* on sigma_z, but rather it's computation can
    reuse sigma_z. So if the actual sigma_z changes analytically, the
    computation in this function might not be correct, e.g. if I make
    x_m conditionally independent given x_o as in expectation_maximisation.py
    Used in analytic EM.
    """
    F_m = F.unsqueeze(0).expand(M.shape[0], -1, -1) * (~M).unsqueeze(-1)

    return F_m @ sigma_z


def exact_inference_np(x, m, F, cov, mean):
    """
    Same implementation as `exact_inference`, but is not batched and
    uses NumPy. Only used to check the torch implementation in
    `exact_inference`.
    """
    # Compute common
    latent_dim = F.shape[1]
    m_inv_cov = m * (1/cov)

    # Calc posterior latent params
    sigma_z = np.linalg.inv(np.eye(latent_dim)
                            + F.T @ (m_inv_cov[:, None] * F)
                            )
    mu_z = sigma_z @ ((m_inv_cov * (x - mean)) @ F)

    # Calc observation posterior
    mu_x, sigma_x = observable_posterior_np(mu_z, sigma_z, F, cov, mean)

    return mu_z, sigma_z, mu_x, sigma_x


# Full Covariance Approximate Inference (FCA)

def fca_inference(X, M, F, cov, mean):
    """
    Computes the posterior distribution parameters of missing X,
    given the observed X. Uses the Full Covariance Approximation (FCA)
    of the latent posterior in the presence of missing variables.
    Args:
        X (N, D): input image batch
        M (N, D): image binary mask, 0 - missing, 1 - observed.
        F: factor loadings
        cov (D): prior diagonal covariance
        mean (D): prior mean
    Returns:
        mu_z: posterior mean of the latents, given the observed values in X
        sigma_z: posterior covariance of the latents, given the observed
            values in X
        mu_x: posterior mean of the observable values, given the observed
            values in X used to find the mean of the missing values
        sigma_x: posterior covariance of the observable values, given the
            observed values in X used to find the distribution of the missing
            values
    """
    # Impute sample with prior mean
    X = X*M + mean*(1-M)

    # Compute common
    latent_dim = F.shape[1]
    inv_cov = (1/cov)

    # Calc posterior latent params
    sigma_z = torch.inverse(torch.eye(latent_dim,
                                      device=X.device,
                                      dtype=X.dtype)
                            + torch.matmul(F.t(), inv_cov.unsqueeze(-1) * F)
                            )
    mu_z = torch.matmul(sigma_z,
                        torch.matmul((inv_cov * (X - mean)), F).unsqueeze(-1)
                        )

    # Calc observation posterior
    mu_x, sigma_x = observable_posterior(mu_z, sigma_z, F, cov, mean)

    # Reshape so that we have a *view* of the covariance for each sample
    sigma_z = sigma_z.unsqueeze(0).expand(X.shape[0],
                                          sigma_z.shape[0],
                                          sigma_z.shape[1])
    sigma_x = sigma_x.unsqueeze(0).expand(X.shape[0],
                                          sigma_x.shape[0],
                                          sigma_x.shape[1])

    return mu_z, sigma_z, mu_x, sigma_x


def fca_inference_np(x, m, F, cov, mean):
    """
    Same implementation `fca_inference`, but is not batched and uses NumPy.
    Only used to check the torch implementation in `fca_inference`.
    """
    # Impute sample with prior mean
    x = x*m + mean*(1-m)

    # Compute common
    latent_dim = F.shape[1]
    inv_cov = 1/cov

    # Calc posterior parameters
    sigma_z = np.linalg.inv(np.eye(latent_dim) + F.T @ (inv_cov[:, None] * F))
    mu_z = sigma_z @ ((inv_cov * (x - mean)) @ F)

    # Calc observation posterior
    mu_x, sigma_x = observable_posterior_np(mu_z, sigma_z, F, cov, mean)

    return mu_z, sigma_z, mu_x, sigma_x


# Scaled Covariance Approximation (SCA)
# TODO: look into SCA efficiency, it's almost as slow as the exact inference
def sca_inference(X, M, F, cov, mean):
    """
    Computes the posterior distribution parameters of missing X, given the
    observed X. Uses the Scaled Covariance Approximation (SCA) of the latent
    posterior in the presence of missing variables.
    Args: X (N, D): input image
        batch M (N, D): image binary mask, 0 - missing, 1 - observed.
        F: factor loadings
        cov (D): prior diagonal covariance
        mean (D): prior mean
    Returns:
        mu_z: posterior mean of the latents, given the observed values in X
        sigma_z: posterior covariance of the latents, given the observed
            values in X
        mu_x: posterior mean of the observable values, given the observed
            values in X used to find the mean of the missing values
        sigma_x: posterior covariance of the observable values, given the
        observed values in X used to find the distribution of the missing
        values
    """
    # Impute sample with prior mean
    X = X*M + mean*(1-M)

    # Compute common
    latent_dim = F.shape[1]
    visible_dim = X.shape[-1]
    inv_cov = (1/cov)
    # Count observed and unobserved dimensions
    c_v = torch.tensor(np.count_nonzero(M, axis=-1)).type_as(X)[:, None]
    c_m = visible_dim - c_v

    # Using the invariance to rotation-of-factors we can transform F
    # by the eigenmatrix of sigma_z
    sigma_z = torch.inverse(torch.eye(latent_dim,
                                      device=X.device,
                                      dtype=X.dtype)
                            + torch.matmul(F.t(), inv_cov.unsqueeze(-1) * F)
                            )
    _, E = torch.eig(sigma_z, eigenvectors=True)
    F_tilde = F @ E

    # Calc latent posterior parameters
    P_z_diag = torch.diagonal(torch.eye(latent_dim)
                              + torch.matmul(F_tilde.t(),
                                             inv_cov.unsqueeze(-1) * F_tilde))
    sigma_z = torch.diag_embed(1/(c_m/visible_dim + c_v/visible_dim*P_z_diag))
    mu_z = torch.matmul(sigma_z,
                        torch.matmul((inv_cov * (X - mean)), F_tilde)
                        .unsqueeze(-1)
                        )

    # Calc observation posterior
    # TODO: We may be able to improve efficiency here by using the fact
    # that sigma_z is diagonal
    mu_x, sigma_x = observable_posterior(mu_z, sigma_z, F_tilde, cov, mean)

    return mu_z, sigma_z, mu_x, sigma_x


def sca_inference_np(x, m, F, cov, mean):
    """
    Same implementation `sca_inference`, but is not batched and uses NumPy.
    Only used to check the torch implementation in `sca_inference`.
    """
    # Impute sample with prior mean
    x = x*m + mean*(1-m)

    # Compute common
    latent_dim = F.shape[1]
    visible_dim = x.shape[-1]
    inv_cov = 1/cov
    # Count observed and unobserved dimensions
    c_v = np.count_nonzero(m)
    c_m = visible_dim - c_v

    # Using the invariance to rotation-of-factors we can transform F
    # by the eigenmatrix of sigma_z
    sigma_z = np.linalg.inv(np.eye(latent_dim) + F.T @ (inv_cov[:, None] * F))
    _, E = np.linalg.eig(sigma_z)
    F_tilde = F @ E

    # Compute latent posterior parameters
    P_z_diag = np.diag(np.eye(latent_dim)
                       + F_tilde.T @ (inv_cov[:, None] * F_tilde))
    sigma_z = np.diag(1/(c_m/visible_dim + c_v/visible_dim*P_z_diag))
    mu_z = sigma_z @ ((inv_cov * (x - mean)) @ F_tilde)

    # Calc observation posterior
    mu_x, sigma_x = observable_posterior_np(mu_z, sigma_z, F_tilde, cov, mean)

    return mu_z, sigma_z, mu_x, sigma_x


#
# Finding the posterior of the observables given the posterior of latents
#
def observable_posterior(mean_z, cov_z, F, cov, mean):
    """
    Computes the posterior distribution of the visibles, given the
    posterior of the latents.
    """
    mean_x = torch.matmul(F, mean_z).squeeze(-1) + mean
    cov_x = torch.matmul(F, torch.matmul(cov_z, F.t())) + torch.diag(cov)
    return mean_x, cov_x


def observable_posterior_np(mean_z, cov_z, F, cov, mean):
    """
    Computes the posterior distribution of the visibles, given the posterior
    of the latents.
    """
    mean_x = F @ mean_z + mean
    cov_x = F @ cov_z @ F.T + np.diag(cov)
    return mean_x, cov_x


#
# Gaussian conditional parameters derivation from joint parameters
#

def univar_gaussian_conditional_params(X, dim, mean, cov):
    # Index for batch dim
    n = torch.arange(X.shape[0])
    # Index for feature dim to reference rows/cols that are not in dim
    i = torch.arange(0, cov.shape[-1])[None, :].repeat(X.shape[0], 1)
    i = i[i != dim[:, None]].reshape(X.shape[0], -1)

    # Exclude current dimension
    X = X[n[:, None], i]

    # Find covariance submatrices
    #A = cov[n, dim, dim]
    B_inv = torch.inverse(cov[n[:, None, None], i[:, :, None], i[:, None, :]])
    C = cov[n[:, None], dim[:, None], i]

    mean_cond = mean[n, dim] \
        + torch.matmul(C.unsqueeze(1),
                       torch.matmul(B_inv,
                                    (X - mean[n[:, None], i]).unsqueeze(-1))
                       ).squeeze(-1).squeeze()
    # cov_cond = A - torch.matmul(C.unsqueeze(1),
    #                             torch.matmul(B_inv, C.unsqueeze(-1)))\
    #   .squeeze()
    # Shur's complement is faster
    # (https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions)
    cov_cond = 1/torch.inverse(cov)[n, dim, dim]

    return mean_cond, cov_cond


def univar_gaussian_conditional_params_np(x, dim, mean, cov):
    # Create index to reference *not* _dim_ rows/columns
    i = np.arange(cov.shape[0])
    i = i[i != dim]

    # Exclude current dimension
    x = x[i]

    # Find covariance submatrices
    #A = cov[dim, dim]
    B_inv = np.linalg.inv(cov[i[:, None], i])
    C = cov[dim, i]

    mean_cond = mean[dim] + C @ B_inv @ (x - mean[i])
    # cov_cond = A - C @ B_inv @ C.T
    # Shur's complement is faster
    # (https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions)
    cov_cond = 1/np.linalg.inv(cov)[dim, dim]

    return mean_cond, cov_cond


def univar_gaussian_conditional_params_from_joint(X, M, m_indices, mean, cov):
    means = torch.empty(X.shape[0])
    covs = torch.empty(X.shape[0])
    # We need to loop here, since the marginal mean/covariance matrix size
    # depends on the pattern of missigness and therefore cannot be easily batch
    # computed
    for i in range(X.shape[0]):
        # Find marginal of the observed and the selected missing variable
        observed_idx = np.argwhere(M[i, :] == 1)
        marginal_idx = np.insert(observed_idx, 0, m_indices[i])
        mean_x_marg = mean[i, marginal_idx]
        cov_x_marg = cov[i, marginal_idx[:, None], marginal_idx]

        # Compute the univariate conditinal params of the missing variable
        # given the observed
        means[i], covs[i] = univar_gaussian_conditional_params(
                                X[i].unsqueeze(0),
                                # Missing variable is at index 0
                                dim=torch.tensor(0).unsqueeze(0),
                                mean=mean_x_marg.unsqueeze(0),
                                cov=cov_x_marg.unsqueeze(0))

    return means, covs
