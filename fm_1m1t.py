import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from forecast_models import ForecastModelABC
from forecast_metadata import QueryTemplateMD, ForecastMD
from constants import TXN_AWARE_PARAM_NEW_VAL_TOKEN, SCHEMA_INT, SCHEMA_TIMESTAMP
from deepar.dataloader import ParamDataset
from deepar.deepar import DeepAR

from typing import Tuple
from collections import defaultdict


import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# DeepAR config
EPOCHS = 60  # TODO: 1 epoch for debug
LR = 0.00005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EARLY_STOPPING_TOLERANCE = 4

LSTM_DROPOUT = 0.1
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 4
EMBEDDING_DIM = 32

# Constants
MODEL_SAVE_PATH = "./artifacts/models/1m1t/"


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=LSTM_HIDDEN_DIM, num_layers=LSTM_LAYERS):
        super(Network, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, dropout=0.1
        )
        self.classification = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=output_size),
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        # output: (L, 1 * H_out)

        out = self.classification(output)
        return out


class Jackie1m1t(ForecastModelABC):
    def __init__(
        self,
        prediction_interval=pd.Timedelta("1S"),
        total_window_size=10,
        history_window_size=8,
        prediction_window_size=2,
        stride_size=1,
    ):
        """
        Parameters
        ----------
        prediction_interval : pd.Timedelta
            Prediction interval refers to the bucket width when aggregating data.
            e.g., if the interval is two seconds, then data is aggregated into two second intervals.
        total_window_size : int
            Total window size refers to the total number of consecutive buckets used in a prediction.
            A window is divided into two parts, history window and prediction window. The history
            window contains historical data and will provide context for prediction. The prediction
            window size are the number of predictions needed to be made based on historical data.
            Note that total_window_size = history_window_size + prediction_window_size
            During training, we have access to data for all total_window_size buckets, whereas in
            testing, we only have access to historical data within history_window_size buckets.

            e.g., total_window_size=10 means there are 10 buckets in total, history_window_size=8
            indicates 8 of the 10 buckets are recorded historical data, prediction_window_size=2
            indicates predictions need to be made for the last 2 of the 10 buckets.
            If there are fewer than five buckets available, we pad-before with zeros.
        history_window_size : int
            Specify the number of buckets containing historical data.
        prediction_window_size : int
            Specify how many predictions need to be made.
        stride_size : int
        """
        assert history_window_size + prediction_window_size == total_window_size, "window sizes don't match"

        # Quantiles to be used to generate training data.
        # quantiles_def is of the form (quantile_name, quantile_val).
        self.quantiles_def = [
            (0, 0.01),
            (10, 0.1),
            (20, 0.2),
            (30, 0.3),
            (40, 0.4),
            (50, 0.5),
            (60, 0.6),
            (70, 0.7),
            (80, 0.8),
            (90, 0.9),
            (100, 0.99),
        ]

        # Prediction interval hyperparameters
        self.prediction_interval = prediction_interval
        self.total_window_size = total_window_size
        self.history_window_size = history_window_size
        self.prediction_window_size = prediction_window_size
        self.stride_size = stride_size

        self._rng = np.random.default_rng(seed=15799)

    def fit(self, forecast_md):
        model = {}
        for qt_enc, qtmd in tqdm(
            forecast_md.qtmds.items(), total=len(forecast_md.qtmds), desc="Fitting query templates."
        ):

            qt = qtmd._query_template
            print(f"Fitting parameters for: {qt}...")

            # Build one DeepAR model per query template
            model[qt] = {}
            model[qt]["params_md"] = {}

            qt_X_train = []  # (N, window_size, num_quantiles + num_covariates)
            qt_X_test = []
            qt_Y_train = []  # (N, window_size, num_quantiles)
            qt_Y_test = []
            # Every parameter that will be used in model training is associated with a class index.
            # This index will be used as an input feature to uniquely identify the parameter.
            curr_class_index = 0
            qt_class_index_train = []  # (N, ) --> each window has a class_index
            qt_class_index_test = []
            # DeepAR scales data on a per-window basis. These arrays store the scaling factors
            # (mean and std) for every window
            qt_scaling_factors_train = []  # (N, 2)
            qt_scaling_factors_test = []

            # The historical parameter data is obtained in the form of a dataframe,
            # where each column corresponds to a different parameter.
            # For example, $1 would map to the first column, and $2 would map to the second column.
            params_df = qtmd.get_historical_params().copy(deep=True)

            for param_idx, param_col in tqdm(
                enumerate(params_df, 1),
                total=len(params_df.columns),
                leave=False,
            ):
                model[qt]["params_md"][param_idx] = {}

                params: pd.Series = params_df[param_col]
                # If the parameter is of string type, we store the values and will sample from them later.
                if str(params.dtype) == "string":
                    model[qt]["params_md"][param_idx]["type"] = "sample"
                    model[qt]["params_md"][param_idx]["sample"] = params
                    continue

                # Convert datetime64 type parameter to numerical value (# of seconds)
                if str(params.dtype) == "datetime64[ns]":
                    params_df[param_col] = params_df[param_col].values.astype("float64")

                # The parameter values are then used to create our X and Y vectors for prediction.
                # The parameter values are bucketed into `prediction_interval` sized buckets.
                # Within these buckets, the quantile values for all the specified `quantiles` functions are computed.
                # Each data point is of the form [ quantile_1, quantile_2, ..., quantile_n ].
                # This creates an initial dataset with N rows and num_quantiles columns,
                # i.e., tsdf has shape (N, num_quantiles),
                # where N is governed by the historical data available and the resampling window `prediction_interval`,
                # and num_quantiles is controlled by the `quantiles` functions used.
                # Note: Need optional arg in lambda function or else all lambda functions will be referring to the last
                # qval value
                quantiles = {
                    qname: (lambda x, curr_qval=qval: x.quantile(curr_qval)) for qname, qval in self.quantiles_def
                }
                tsdf = (
                    params_df[param_col]
                    .resample(self.prediction_interval)
                    .agg(quantiles)
                    .fillna(method="ffill")
                    .astype("float64")
                )

                # Duplicate if not enough data. Ensure at least one data point in history window to train model
                # eg. [0, 0, 0, ..., 0, 5 | 6, 7, 8]
                #              history        pred     --> at least one point '5' in history window
                min_data_needed = self.prediction_window_size + 2
                num_new_data = min_data_needed - len(tsdf)
                if num_new_data > 0:
                    last_time = tsdf.index[-1]
                    new_index = pd.date_range(
                        last_time + self.prediction_interval, freq=self.prediction_interval, periods=num_new_data
                    )
                    tsdf = pd.concat([tsdf, pd.DataFrame(index=new_index)])
                    tsdf = tsdf.ffill()

                # Additionally, add current timestamp as an input feature. This is called known covariate since it is
                # known at both historical range and prediction range.
                # More known covariates can be added, such as weekday, hour, month, etc.
                tsdf["datetime_ns"] = tsdf.index.values.astype("float64")

                # The timstamp feature is standard-scaled across the entire time range.
                timestamp_mean = tsdf["datetime_ns"].mean()
                timestamp_std = tsdf["datetime_ns"].std()
                tsdf["datetime_ns"] = (tsdf["datetime_ns"] - timestamp_mean) / timestamp_std

                # DeepAR trains a model that uses data at current timestamp to predict data at next timestamp. This can
                # be unrolled arbitrary times to predict arbitrary number of points. We need 1 more data point than
                # total_window_size when computing rolling window.
                # eg. data=[1, 2, 3, 4, 5, ...], total_window_size=4, history_window_size=2, predict_window_size=2
                # Use 1 to predict 2, 2 to 3, etc. we have
                # 2  3  |  4  5  Historical data (X) = [1, 2, 3, 4]
                # |  |  |  |  |  Known labels (Y) = [2, 3, 4, 5]
                # 1  2  |  3  4  Therefore data [1, 2, 3, 4, 5] are needed to construct X and Y
                # TODO: rolling with stride
                X, Y, scaling_factors = [], [], []
                for ridx, rolling_window in enumerate(tsdf.rolling(window=self.total_window_size + 1)):
                    # When padding, ensure there exists at least one data point in history range, ie. if
                    # prediction_window_range = 2, then the rolling window contains at least 3 data points.
                    # Skip windows with less than prediction_window_size data points.
                    if ridx <= self.prediction_window_size:
                        continue

                    # Compute number of rows in current window that belongs to history range, not prediction range,
                    # since scaling only depends on data in the history range, not on the prediction range.
                    num_history_range = min(ridx - self.prediction_window_size, self.history_window_size)

                    # Extract data in history range and compute mean/std based on this subset.
                    rw_history = rolling_window.iloc[:num_history_range]

                    # We want to compute mean/std based on the actual data in params_df[param_col] instead of
                    # on the quantiled data. Use the start time and end time of rolling_history_window to
                    # index into params_df[param_col] and compute mean/var.
                    rw_starttime, rw_endtime = rw_history.index[0], rw_history.index[-1] + self.prediction_interval
                    rw_mean = params_df[param_col][rw_starttime:rw_endtime].mean()
                    rw_std = params_df[param_col][rw_starttime:rw_endtime].std()

                    # Hack: if std==None (only 1 data point) or std=0 (all data are the same), then set it to 1 so that
                    # when scaling/unscaling data, we can simply divide/multiply by std and not encounter divide by 0
                    # error.
                    if pd.isnull(rw_std) or rw_std == 0:
                        rw_std = 1

                    # Create X and Y windows, and scale both windows by scaling factors computed above
                    rolling_window = rolling_window.values
                    curr_X = rolling_window[:-1, :].copy()
                    # When scaling, only scale the quantile data because the covariates have already been scaled
                    curr_X[:, : len(self.quantiles_def)] = (curr_X[:, : len(self.quantiles_def)] - rw_mean) / rw_std
                    # We only need the quantile values for Y, not the known covariates (timestamp feature)
                    curr_Y = rolling_window[1:, : len(self.quantiles_def)].copy()
                    curr_Y[:, : len(self.quantiles_def)] = (curr_Y[:, : len(self.quantiles_def)] - rw_mean) / rw_std

                    # Pad the windows to total_window_size if needed
                    num_pad_before = self.total_window_size + 1 - rolling_window.shape[0]
                    if num_pad_before > 0:
                        pad_width = ((num_pad_before, 0), (0, 0))
                        curr_X = np.pad(curr_X, pad_width)
                        curr_Y = np.pad(curr_Y, pad_width)

                        # The last column of X is current timestamp. Therefore
                        # pad this column with previous timestamps instead of 0.
                        min_time = tsdf.index[0]
                        for j in range(1, num_pad_before + 1):
                            prev_time = min_time - j * self.prediction_interval
                            prev_time_scaled = (
                                prev_time.to_datetime64().astype("float64") - timestamp_mean
                            ) / timestamp_std
                            # Apply scaling
                            curr_X[num_pad_before - j, -1] = prev_time_scaled

                    X.append(curr_X)
                    Y.append(curr_Y)
                    scaling_factors.append(np.array([rw_mean, rw_std]))

                X = np.array(X)  # (N, total_window_size, num_quantiles + num_covariates)
                Y = np.array(Y)  # (N, total_window_size, num_quantiles)
                scaling_factors = np.array(scaling_factors)  # (N, 2)
                curr_class_index_arr = np.full(
                    X.shape[0], curr_class_index
                )  # All data in X have the same class index -> (N, )

                # Split data into train test sets
                if len(X) <= 1:
                    print(f"Warning: not enough data ({X.shape=}), will double up rows: {qt}.")
                    X = np.concatenate([X, X])
                    Y = np.concatenate([Y, Y])
                    curr_class_index_arr = np.concatenate([curr_class_index_arr, curr_class_index_arr])
                    scaling_factors = np.concatenate([scaling_factors, scaling_factors])
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X,
                    Y,
                    shuffle=False,
                    test_size=0.1,
                )
                split_index = X_train.shape[0]
                curr_class_index_arr_train = curr_class_index_arr[:split_index]
                curr_class_index_arr_test = curr_class_index_arr[split_index:]
                scaling_factors_train = scaling_factors[:split_index]
                scaling_factors_test = scaling_factors[split_index:]

                qt_X_train.append(X_train)
                qt_X_test.append(X_test)
                qt_Y_train.append(Y_train)
                qt_Y_test.append(Y_test)
                qt_class_index_train.append(curr_class_index_arr_train)
                qt_class_index_test.append(curr_class_index_arr_test)
                qt_scaling_factors_train.append(scaling_factors_train)
                qt_scaling_factors_test.append(scaling_factors_test)

                # Save data for future use
                model[qt]["params_md"][param_idx]["type"] = "jackie1m1t"
                model[qt]["params_md"][param_idx]["jackie1m1t"] = {}
                model[qt]["params_md"][param_idx]["jackie1m1t"]["param_df"] = params_df[param_col]
                model[qt]["params_md"][param_idx]["jackie1m1t"]["quantile_df"] = tsdf
                model[qt]["params_md"][param_idx]["jackie1m1t"]["scaling_factors"] = scaling_factors
                model[qt]["params_md"][param_idx]["jackie1m1t"]["class_index"] = curr_class_index
                model[qt]["params_md"][param_idx]["jackie1m1t"]["timestamp_scaling_factors"] = [
                    timestamp_mean,
                    timestamp_std,
                ]

                # Add 1 to class index
                curr_class_index += 1

            # Here we combine training/testing data for all parameters and train the model
            if len(qt_X_train) == 0:
                # No parameter data for this query template, skip
                continue
            qt_X_train = np.concatenate(qt_X_train, axis=0)
            qt_X_test = np.concatenate(qt_X_test, axis=0)
            qt_Y_train = np.concatenate(qt_Y_train, axis=0)
            qt_Y_test = np.concatenate(qt_Y_test, axis=0)
            qt_class_index_train = np.concatenate(qt_class_index_train, axis=0)
            qt_class_index_test = np.concatenate(qt_class_index_test, axis=0)
            qt_scaling_factors_train = np.concatenate(qt_scaling_factors_train, axis=0)
            qt_scaling_factors_test = np.concatenate(qt_scaling_factors_test, axis=0)

            train_set = ParamDataset(qt_X_train, qt_Y_train, qt_class_index_train, qt_scaling_factors_train)
            test_set = ParamDataset(qt_X_test, qt_Y_test, qt_class_index_test, qt_scaling_factors_test)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

            deepar_model = DeepAR(
                len(self.quantiles_def),
                1,
                LSTM_HIDDEN_DIM,
                LSTM_LAYERS,
                LSTM_DROPOUT,
                EMBEDDING_DIM,
                curr_class_index + 1,  # Total # parameters
                DEVICE,
            ).to(DEVICE)
            optimizer = optim.Adam(deepar_model.parameters(), lr=LR)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.75, patience=1, verbose=True, threshold=1e-2
            )

            # Early stopping: if vloss does not decrease for multiple iterations, stop training
            best_vloss = float("inf")
            best_model = None
            increase_vloss_count = 0
            for epoch in range(EPOCHS):
                train_loss = self._train_epoch(deepar_model, train_loader, optimizer, criterion)
                val_loss = self._validate(deepar_model, test_loader, criterion)
                print(
                    f"Epoch {epoch+1}/{EPOCHS}: Train loss {train_loss:.04f}, Val loss {val_loss:.04f} lr {optimizer.param_groups[0]['lr']:.04f}"
                )

                if val_loss < best_vloss:
                    increase_vloss_count = 0
                    best_vloss = val_loss
                    best_model = deepcopy(deepar_model)
                else:
                    increase_vloss_count += 1
                    if increase_vloss_count == EARLY_STOPPING_TOLERANCE:
                        break
                scheduler.step(val_loss)
                torch.cuda.empty_cache()
            model[qt]["model"] = best_model
        self.model = model
        return self

    def generate_parameters(self, query_template, timestamp):
        target_timestamp = pd.Timestamp(timestamp)

        # Generate the parameters.
        params = {}
        for param_idx in self.model[query_template]:
            fit_obj = self.model[query_template][param_idx]
            fit_type = fit_obj["type"]

            param_val = None
            if fit_type == "sample":
                param_val = self._rng.choice(fit_obj["sample"])
            else:
                param_model = fit_obj["jackie1m1p"]["model"]
                param_mean = fit_obj["jackie1m1p"]["mean"]
                param_std = fit_obj["jackie1m1p"]["std"]
                param_X = fit_obj["jackie1m1p"]["X"]
                param_X_ts = fit_obj["jackie1m1p"]["X_ts"]

                start_timestamp_idx = -1
                # If the start timestamp was within in the training data, use the last value before said timestamp.
                candidate_indexes = np.argwhere(param_X_ts <= target_timestamp)
                if candidate_indexes.shape[0] != 0:
                    start_timestamp_idx = candidate_indexes.max()
                start_timestamp = param_X_ts[start_timestamp_idx]

                # seq : (seq_len, num_quantiles)
                seq = param_X[start_timestamp_idx]
                # seq : (1, seq_len, num_quantiles)
                seq = seq[None, :, :]
                # seq : (seq_len, 1, num_quantiles)
                seq = np.transpose(seq, (1, 0, 2))
                seq = torch.tensor(seq).to(DEVICE).float()

                # TODO(WAN): cache predicted values.

                # Predict until the target timestamp is reached.
                pred = seq[-1, -1, :]
                pred = torch.cummax(pred, dim=0).values
                num_predictions = int((target_timestamp - start_timestamp) / self.prediction_interval)
                for _ in range(num_predictions):
                    # Predict the quantiles from the model.
                    with torch.no_grad():
                        pred = param_model(seq)

                    # Ensure prediction quantile values are strictly increasing.
                    pred = pred[-1, -1, :]
                    pred = torch.cummax(pred, dim=0).values

                    # Add pred to original seq to create new seq for next time stamp.
                    seq = torch.squeeze(seq, axis=1)
                    seq = torch.cat((seq[1:, :], pred[None, :]), axis=0)
                    seq = seq[:, None, :]

                pred = pred.cpu().detach().numpy()

                # Un-normalize the quantiles.
                if param_std != 0:
                    pred = pred * param_std + param_mean
                else:
                    pred = pred + param_mean

                # TODO(WAN): We now have all the quantile values. How do we sample from them?
                # Randomly pick a bucket, and then randomly pick a value.
                # There are len(pred) - 1 many buckets.
                bucket = self._rng.integers(low=0, high=len(pred) - 1, endpoint=False)
                left_bound, right_bound = pred[bucket], pred[bucket + 1]
                param_val = self._rng.uniform(left_bound, right_bound)
            assert param_val is not None

            # Param dict values must be quoted for consistency.
            params[f"${param_idx}"] = f"'{param_val}'"
        return params

    def generate_parameters_txn_aware(
        self, query_template, query_template_encoding, timestamp, transition_params, db_schema, sample_path
    ):
        target_timestamp = pd.Timestamp(timestamp)
        param_data_types = self.get_parameter_data_types(db_schema, query_template)

        # Dict that maps how many times each query template appears in sample_path. This is used
        # to filter the parameter transition dict, since a parameter might depend on many different
        # params that appear before it
        qt_encs_count = defaultdict(int)
        for _, _, _, _, qt_enc in sample_path:
            qt_encs_count[qt_enc] += 1

        # Generate the parameters.
        params = {}
        for param_idx in self.model[query_template]["params_md"]:
            fit_obj = self.model[query_template]["params_md"][param_idx]
            fit_type = fit_obj["type"]

            param_val = None

            if fit_type == "sample":
                param_val = self._rng.choice(fit_obj["sample"])
                assert param_val is not None
                # Param dict values must be quoted for consistency.
                params[f"${param_idx}"] = f"'{param_val}'"
                continue

            # Try to get a parameter value by using transition dict (which encodes dependencies between parameters)
            param_val = self._get_parameter_from_transition_dict(
                query_template_encoding, transition_params, sample_path, param_idx, qt_encs_count
            )
            if param_val != None:
                params[f"${param_idx}"] = param_val  # Previous param is already quoted. No need to quote again
                continue

            # param_val is None, meaning a new value should be generated from the forecast model
            qt_model = self.model[query_template]["model"]
            param_val = self._get_parameter_from_forecast_model(qt_model, fit_obj, target_timestamp)
            assert param_val is not None

            # Cast param_val to corresponding data type
            if param_data_types[param_idx - 1] == SCHEMA_INT:
                param_val = round(param_val)
            elif param_data_types[param_idx - 1] == SCHEMA_TIMESTAMP:
                param_val = pd.to_datetime(param_val, format="%Y-%m-%d %H:%M:%S.%f")

            # Param dict values must be quoted for consistency.
            params[f"${param_idx}"] = f"'{param_val}'"

        return params

    def _get_parameter_from_transition_dict(
        self, query_template_encoding, transition_params, sample_path, param_idx, qt_encs_count
    ):
        """Get a parameter value by examining what previous parameters it depends on. Return None if
        no dependency is found, and a value should be generated from the forecast model.

        Args:
            query_template_encoding (int)
            transition_params (Dict): Contains dependencies between all parameters
            sample_path (List): all queries with parameters generated so far
            param_idx (int): index of the parameter in which value is to be sampled
            qt_encs_count (Dict): Map each qt_enc to how many times it appears in sample path

        Returns:
            str: value of parameter, None if no dependency found
        """
        # Extract dependencies of current parameters wrt all parameters seen in sample_path
        qtp_enc = f"{query_template_encoding}_{param_idx}"

        # Current parameter does not exist in transition dict
        if qtp_enc not in transition_params:
            return None

        candidate_qtp_enc_most_recent = transition_params[qtp_enc].keys()
        final_candidates = []
        final_probs = []

        # Check which entries in transition dict actually appear in the sample_path
        for qtp_enc_most_recent in candidate_qtp_enc_most_recent:
            # The NEW_VAL_TOKEN entry indicates that a new value is to be sampled from the
            # model. This is always included in the transition dict.
            if qtp_enc_most_recent == TXN_AWARE_PARAM_NEW_VAL_TOKEN:
                final_candidates.append(qtp_enc_most_recent)
                final_probs.append(transition_params[qtp_enc][qtp_enc_most_recent])
                continue

            # qtp_enc_most_recent --> {qt_enc}_{param_index}_{nth most recently seen}
            splits = qtp_enc_most_recent.split("_")
            qt_enc = int(splits[0])
            nth_most_recent = int(splits[2])
            if qt_enc in qt_encs_count and nth_most_recent <= qt_encs_count[qt_enc]:
                final_candidates.append(qtp_enc_most_recent)
                final_probs.append(transition_params[qtp_enc][qtp_enc_most_recent])

        # If current parameter does not depend on any previous parameter, then sample a new one
        if len(final_candidates) == 0:
            return None

        final_probs = np.array(final_probs)
        final_probs = final_probs / np.sum(final_probs)
        qtp_enc_most_recent = np.random.choice(final_candidates, p=final_probs)

        # Selected transition is NEW_VAL_TOKEN, meaning a new value is to be sampled
        if qtp_enc_most_recent == TXN_AWARE_PARAM_NEW_VAL_TOKEN:
            return None

        splits = qtp_enc_most_recent.split("_")
        dst_qt_enc = int(splits[0])
        dst_param_idx = int(splits[1])
        nth_most_recent = int(splits[2])

        # Locate the nth most recently seen `dst_qt_enc`, and retrieve its `dst_param_idx` param.
        seen_times = 0
        for _, _, _, qt_params, qt_enc in reversed(sample_path):
            if qt_enc == dst_qt_enc:
                seen_times += 1

            if nth_most_recent == seen_times:
                param_val = qt_params[f"${dst_param_idx}"]
                break
        assert param_val is not None

        return param_val

    def _get_parameter_from_forecast_model(self, model, fit_obj, target_timestamp):
        """Generate a parameter value from the corresponding fit_obj, by continously unrolling
        RNN until target_timestamp.
        """
        param_df = fit_obj["jackie1m1t"]["param_df"]
        quantile_df = fit_obj["jackie1m1t"]["quantile_df"]
        quantile_timestamp = quantile_df.index
        params_ts_scaling_factors = fit_obj["jackie1m1t"]["timestamp_scaling_factors"]  # (2, )
        timestamp_mean, timestamp_std = params_ts_scaling_factors[0], params_ts_scaling_factors[1]
        params_class_index = fit_obj["jackie1m1t"]["class_index"]

        assert (len(quantile_timestamp) > 0, "Must have at least 1 historical data point when generating data")

        # Get start and end timestamp of a training window.
        # If the start timestamp was within in the training data, use the last value before said timestamp.
        window_end_idx = len(quantile_timestamp) - 1
        candidate_indexes = np.argwhere(quantile_timestamp <= target_timestamp)
        if candidate_indexes.shape[0] != 0:
            window_end_idx = candidate_indexes.max()
        window_start_idx = max(0, window_end_idx - self.history_window_size + 1)
        window_start_ts = quantile_timestamp.to_numpy()[window_start_idx]
        window_end_ts = quantile_timestamp.to_numpy()[window_end_idx] + self.prediction_interval

        # Retrieve `history_window_size` data points before `start_timestamp`. This is the data in training window
        # The quantile data are not scaled, but known covariates (timestamps) are scaled
        # (history_window_size, quantile_dim+cov_dim)
        seq = quantile_df[window_start_ts:window_end_ts].to_numpy()

        assert (
            seq.shape[0] == self.history_window_size,
            "Wrong number of data points in window when genrating predictions",
        )

        # Compute window mean/std using data from param_df. Then scale the quantile part of the window
        # using these factors.
        window_mean = param_df[window_start_ts:window_end_ts].mean()
        window_std = param_df[window_start_ts:window_end_ts].std()
        if pd.isnull(window_std) or window_std == 0:
            window_std = 1
        seq[:, : len(self.quantiles_def)] = (seq[:, : len(self.quantiles_def)] - window_mean) / window_std

        # Pad the window to `history_window_size` if needed
        # Pad the windows to total_window_size if needed
        num_pad_before = max(0, self.history_window_size - seq.shape[0])
        pad_width = ((num_pad_before, 0), (0, 0))
        seq = np.pad(seq, pad_width)

        # We want to make `seq` to have shape (total_window_size, quantile_dim+cov_dim).
        # Pad seq with `num_predictions` rows of 0's.
        num_predictions = int(
            (target_timestamp - (window_end_ts - self.prediction_interval)) / self.prediction_interval
        )
        pad_width = ((0, num_predictions), (0, 0))
        seq = np.pad(seq, pad_width)
        # seq : (total_window_size, quantile_dim+cov_dim)

        # The last column of seq should be prefilled with timestamp data instead of padded with 0's
        entire_window_start_ts = window_start_ts - num_pad_before * self.prediction_interval
        total_window_size = self.history_window_size + num_predictions
        entire_window_ts = pd.date_range(
            start=entire_window_start_ts, periods=total_window_size, freq=self.prediction_interval
        ).values.astype("float64")
        entire_window_ts_scaled = (entire_window_ts - timestamp_mean) / timestamp_std
        seq[:, -1] = entire_window_ts_scaled

        # seq : (1, seq_len, num_quantiles+cov_dim)
        seq = torch.tensor(seq[None, :, :]).to(DEVICE).float()

        # Unroll LSTM in training window, then in testing window
        B, T = seq.shape[0], seq.shape[1]
        seq = seq.permute(1, 0, 2).float().to(DEVICE)  # (1, T, quantile+cov_dim) -> (T, 1, quantile+cov_dim)
        class_ids = torch.tensor([[params_class_index]]).to(DEVICE)  # (1, 1)

        hidden, cell = model.init_hidden(B), model.init_cell(B)
        for t in range(self.history_window_size):
            quantiles, hidden, cell = model(seq[t].unsqueeze(0), class_ids, hidden, cell)
        pred_quantiles = model.test(
            seq, class_ids, hidden, cell, self.history_window_size, num_predictions
        )  # (pred_window_size, 1, q_dim)

        # Un-normalize the quantiles.
        pred_quantiles = (pred_quantiles.squeeze() * window_std) + window_mean
        pred_quantiles = pred_quantiles[-1, :]
        pred_quantiles = torch.cummax(pred_quantiles, dim=0).values.cpu().detach().numpy()

        # TODO(WAN): We now have all the quantile values. How do we sample from them?
        # Randomly pick a bucket, and then randomly pick a value.
        # There are len(pred) - 1 many buckets.
        bucket = self._rng.integers(low=0, high=len(pred_quantiles) - 1, endpoint=False)
        left_bound, right_bound = pred_quantiles[bucket], pred_quantiles[bucket + 1]
        param_val = self._rng.uniform(left_bound, right_bound)

        return param_val

    ###################################################################################################
    #########################           Model Training        #########################################
    ###################################################################################################

    def _train_epoch(self, model, loader, optimizer, criterion):
        model.train()

        batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

        running_loss = 0
        for i, (zxs, class_ids, _, labels) in enumerate(loader):
            optimizer.zero_grad()

            B, T = zxs.shape[0], zxs.shape[1]

            zxs = zxs.permute(1, 0, 2).float().to(DEVICE)  # (B, T, quantile+cov_dim) -> (T, B, quantile+cov_dim)
            class_ids = class_ids.unsqueeze(0).to(DEVICE)  # (B) -> (1, B)
            labels = labels.permute(1, 0, 2).float().to(DEVICE)  # (B, T, quantile) -> (T, B, quantile)

            loss = 0
            hidden, cell = model.init_hidden(B), model.init_cell(B)
            for t in range(self.total_window_size):
                quantiles, hidden, cell = model(zxs[t].unsqueeze(0), class_ids, hidden, cell)
                loss += criterion(quantiles, labels[t])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batch_bar.set_postfix(loss="{:.04f}".format(running_loss / (i + 1)))
            batch_bar.update()

        batch_bar.close()
        return running_loss / len(loader)

    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion,
    ):
        model.eval()

        batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Eval")

        running_loss = 0
        with torch.no_grad():
            for i, (zxs, class_ids, scaling_factors, labels) in enumerate(loader):
                B = zxs.shape[0]

                zxs = zxs.permute(1, 0, 2).float().to(DEVICE)  # (T, B, q_dim+cov_dim)
                class_ids = class_ids.unsqueeze(0).to(DEVICE)  # (1, B)
                scaling_factors = scaling_factors.to(DEVICE)  # (B, 2)
                labels = labels.permute(1, 0, 2).float().to(DEVICE)  # (T, B, q_dim)

                scaled_quantiles = torch.zeros((self.history_window_size, B, len(self.quantiles_def)), device=DEVICE)
                hidden, cell = model.init_hidden(B), model.init_cell(B)
                # Compute hidden state up until all points in history window
                for t in range(self.history_window_size):
                    quantiles, hidden, cell = model(zxs[t].unsqueeze(0), class_ids, hidden, cell)  # (B, q_dim)
                    # scale: (B, q_dim) * (B, 1)
                    scaled_quantiles[t] = (quantiles + scaling_factors[:, 0].unsqueeze(1)) * scaling_factors[
                        :, 1
                    ].unsqueeze(1)

                pred_quantiles = model.test(
                    zxs, class_ids, hidden, cell, self.history_window_size, self.prediction_window_size
                )  # (pred_window_size, B, q_dim)

                # Scale predicted quantiles and actual quantiles and compute loss
                target_quantiles = labels[self.history_window_size :]  # (pred_window_size, B, q_dim)

                # (pred_window_size, B, q_dim) * (1, B, 1)
                target_quantiles = (target_quantiles + scaling_factors[:, 0][None, :, None]) * scaling_factors[:, 1][
                    None, :, None
                ]
                pred_quantiles = (pred_quantiles + scaling_factors[:, 0][None, :, None]) * scaling_factors[:, 1][
                    None, :, None
                ]

                running_loss += criterion(target_quantiles, pred_quantiles).item()

                batch_bar.set_postfix(loss="{:.04f}".format(running_loss / (i + 1)))
                batch_bar.update()

        batch_bar.close()
        return running_loss / len(loader)


if __name__ == "__main__":
    fmd = ForecastMD.load("fmd.pkl")

    # Fit the forecaster object.
    forecaster = Jackie1m1t()
    forecaster.fit(fmd)

    # query_log_filename = "./preprocessed.parquet.gzip"

    # forecaster = Forecaster(
    #     pred_interval=pd.Timedelta("2S"), pred_seq_len=5, pred_horizon=pd.Timedelta("2S"), load_metadata=False
    # )
    # forecaster.fit(query_log_filename)
    #
    # forecaster = Forecaster(
    #     pred_interval=pd.Timedelta("2S"), pred_seq_len=5, pred_horizon=pd.Timedelta("2S"), load_metadata=True
    # )
    # pred_result = forecaster.get_parameters_for(
    #     "DELETE FROM new_order WHERE NO_O_ID = $1 AND NO_D_ID = $2 AND NO_W_ID = $3",
    #     "2022-03-08 11:30:06.021000-0500",
    #     30,
    # )
    # with np.printoptions(precision=3, suppress=True):
    #     print(pred_result)
    # forecaster.get_all_parameters_for("DELETE FROM new_order WHERE NO_O_ID = $1 AND NO_D_ID = $2 AND NO_W_ID = $3")
