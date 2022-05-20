from abc import abstractmethod, ABC
import re
from constants import SCHEMA_INT


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

    def get_parameter_data_types(self, db_schema, query_template):
        param_data_types = []
        if "INSERT" in query_template:
            # Query: INSERT INTO history (H_D_ID, H_C_W_ID, ...) VALUES ($1, $2, ...)
            # Extract all column names within the first parenthesis
            # Then split by ',' to get each column name
            r = re.compile(r"\(([^\$]+)\)")
            param_cols = r.findall(query_template)[0]
            param_cols = param_cols.split(", ")
        else:
            # UPDATE warehouse SET W_YTD = W_YTD + $1 WHERE W_ID = $2
            # Extract column name such as W_YTD and W_ID from query temmplate
            r = re.compile(r"([\S]+) (?:=|>=|<=|>|<) (?:\S+ \+ )?\$\d+")
            param_cols = r.findall(query_template)

        for param_col_name in param_cols:
            param_data_types.append(db_schema[param_col_name].get_type())
        # For TPCC, LIMIT is always the last parameter in a query.
        # This might change for other benchmarks
        if "LIMIT" in query_template:
            param_data_types.append(SCHEMA_INT)

        return param_data_types
