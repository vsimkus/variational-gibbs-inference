import cdi.submodules.missing_data_provider as mdp


class MissingDataProvider(mdp.MissingDataProvider):
    """
    Overrides original MissingDataProvider to allow mutation of the underlying dataset.
    """
    __doc__ = __doc__ + mdp.MissingDataProvider.__doc__

    def __setitem__(self, key, value):
        """
        Allows mutation of the underlying dataset.
        The missingness mask is immutable.
        """
        self.dataset[key] = value
