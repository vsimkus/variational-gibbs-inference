from collections import namedtuple
from time import time

import numpy as np
import sklearn.impute
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as ItImputer
from sklearn.utils import (_safe_indexing, check_array, check_random_state,
                           is_scalar_nan)

_ImputerTriplet = namedtuple('_ImputerTriplet', ['feat_idx',
                                                 'neighbor_feat_idx',
                                                 'estimator'])


class IterativeImputer(ItImputer):
    """
    Do not store intermediate estimators, because uses HUGE amounts of data.
    """

    def fit_transform(self, X, y=None):
        """Fits the imputer on X and return the transformed X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        self.random_state_ = getattr(self, "random_state_",
                                     check_random_state(self.random_state))

        if self.max_iter < 0:
            raise ValueError(
                "'max_iter' should be a positive integer. Got {} instead."
                .format(self.max_iter))

        if self.tol < 0:
            raise ValueError(
                "'tol' should be a non-negative float. Got {} instead."
                .format(self.tol)
            )

        if self.estimator is None:
            from ..linear_model import BayesianRidge
            self._estimator = BayesianRidge()
        else:
            self._estimator = clone(self.estimator)

        if hasattr(self._estimator, 'random_state'):
            self._estimator.random_state = self.random_state_

        # NOTE: removed this
        # self.imputation_sequence_ = []

        self.initial_imputer_ = None
        super()._fit_indicator(X)
        X_indicator = super()._transform_indicator(X)
        X, Xt, mask_missing_values = self._initial_imputation(X)
        if self.max_iter == 0 or np.all(mask_missing_values):
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        # Edge case: a single feature. We return the initial ...
        if Xt.shape[1] == 1:
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        self._min_value = IterativeImputer._validate_limit(
            self.min_value, "min", X.shape[1])
        self._max_value = IterativeImputer._validate_limit(
            self.max_value, "max", X.shape[1])

        if not np.all(np.greater(self._max_value, self._min_value)):
            raise ValueError(
                "One (or more) features have min_value >= max_value.")

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)
        self.n_features_with_missing_ = len(ordered_idx)

        abs_corr_mat = self._get_abs_corr_mat(Xt)

        n_samples, n_features = Xt.shape
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s"
                  % (X.shape,))
        start_t = time()
        if not self.sample_posterior:
            Xt_previous = Xt.copy()
            normalized_tol = self.tol * np.max(
                np.abs(X[~mask_missing_values])
            )
        for self.n_iter_ in range(1, self.max_iter + 1):
            if self.imputation_order == 'random':
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(n_features,
                                                                feat_idx,
                                                                abs_corr_mat)
                Xt, estimator = self._impute_one_feature(
                    Xt, mask_missing_values, feat_idx, neighbor_feat_idx,
                    estimator=None, fit_mode=True)
                estimator_triplet = _ImputerTriplet(feat_idx,
                                                    neighbor_feat_idx,
                                                    estimator)
                # NOTE: removed this
                # self.imputation_sequence_.append(estimator_triplet)

            if self.verbose > 1:
                print('[IterativeImputer] Ending imputation round '
                      '%d/%d, elapsed time %0.2f'
                      % (self.n_iter_, self.max_iter, time() - start_t))

            if not self.sample_posterior:
                inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf,
                                          axis=None)
                if self.verbose > 0:
                    print('[IterativeImputer] '
                          'Change: {}, scaled tolerance: {} '.format(
                              inf_norm, normalized_tol))
                if inf_norm < normalized_tol:
                    if self.verbose > 0:
                        print('[IterativeImputer] Early stopping criterion '
                              'reached.')
                    break
                Xt_previous = Xt.copy()
        else:
            if not self.sample_posterior:
                warnings.warn("[IterativeImputer] Early stopping criterion not"
                              " reached.", ConvergenceWarning)
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return super()._concatenate_indicator(Xt, X_indicator)
