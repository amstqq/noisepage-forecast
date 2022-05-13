from abc import abstractmethod, ABC


class ForecastModelABC(ABC):
    @abstractmethod
    def fit(self, forecast_md):
        """
        Fit the forecast model to the forecast metadata as of that point in time.

        Parameters
        ----------
        forecast_md : ForecastMD

        Returns
        -------

        """
        pass

    @abstractmethod
    def generate_parameters(self, query_template, timestamp):
        """
        Generate a set of parameters for the specified query template as of the specified time.

        Parameters
        ----------
        query_template : str
        timestamp : str

        Returns
        -------
        params : Dict[str, str]
            The generated parameters, a dict mapping from "$n" keys to quoted parameter values.
        """
        pass

    @abstractmethod
    def generate_parameters_txn_aware(
        self, query_template, query_template_encoding, timestamp, transition_params, sample_path
    ):
        """Generate a set of txn-aware parameters for the specified query template as of the specified time.

        Args:
            query_template (str)
            query_template_encoding (int)
            current_session_ts (str)
            transition_params (Dict): Map dependencies of one parameter to other parameters
            sample_path (List): Previously generated queries
        """
        pass
