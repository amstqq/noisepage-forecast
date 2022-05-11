import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from scipy import stats
import joypy
import matplotlib.pyplot as plt

import pickle
import os
import warnings
import random

from pathlib import Path

from forecast_models import ForecastModelABC
from forecast_metadata import QueryTemplateMD, ForecastMD
from sklearn.model_selection import train_test_split

from typing import Tuple

# LSTM config
HIDDEN_SIZE = 128
RNN_LAYERS = 2
EPOCHS = 50
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
MODEL_SAVE_PATH = "./artifacts/models/1m1p/"


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=HIDDEN_SIZE, num_layers=RNN_LAYERS):
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


class Jackie1m1p(ForecastModelABC):
    def __init__(
        self, prediction_interval=pd.Timedelta("2S"), prediction_seq_len=5, prediction_horizon=pd.Timedelta("2S")
    ):
        """
        Parameters
        ----------
        prediction_interval : pd.Timedelta
            Prediction interval refers to the bucket width when aggregating data.
            e.g., if the interval is two seconds, then data is aggregated into two second intervals.
        prediction_seq_len : int
            Prediction sequence length refers to the number of consecutive buckets used in a prediction.
            e.g., if the sequence length is 5, then the last five buckets are always used.
            If there are fewer than five buckets available, we pad-before with zeros.
        prediction_horizon : pd.Timedelta
            Prediction horizon refers to how far into the future to predict.
            e.g., if the horizon is two seconds, then data until time T is used to predict two seconds after T.
        """

        # Quantiles to be used to generate training data.
        # quantiles_def is of the form (quantile_name, quantile_val).
        self.quantiles_def: Tuple[int, int] = [
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
        self.prediction_seq_len = prediction_seq_len
        self.prediction_horizon = prediction_horizon

        self._rng = np.random.default_rng(seed=15799)

    def fit(self, forecast_md):
        model = {}
        for qt_enc, qtmd in tqdm(
            forecast_md.qtmds.items(), total=len(forecast_md.qtmds), desc="Fitting query templates."
        ):
            qt = qtmd._query_template

            model[qt] = {}
            # The historical parameter data is obtained in the form of a dataframe,
            # where each column corresponds to a different parameter.
            # For example, $1 would map to the first column, and $2 would map to the second column.
            params_df = qtmd.get_historical_params().copy(deep=True)
            # Then, for each column in the parameter dataframe,
            for param_idx, param_col in tqdm(
                enumerate(params_df, 1),
                total=len(params_df.columns),
                desc=f"Fitting parameters for: {qt}",
                leave=False,
            ):
                model[qt][param_idx] = {}

                params: pd.Series = params_df[param_col]
                # If the parameter is of string type, we store the values and will sample from them later.
                if str(params.dtype) == "string":
                    model[qt][param_idx]["type"] = "sample"
                    model[qt][param_idx]["sample"] = params
                    continue
                # Otherwise, we normalize the historical parameter values based on StandardScaler normalization.
                # TODO(WAN):
                #  If we use StandardScaler normalization, which this is, then it won't adapt to new min/max.
                #  Could we get away with a different transform?
                mean = params.mean()
                std = params.std()
                if std != 0:
                    normalized = (params - mean) / std
                else:
                    normalized = params - mean
                params_df[param_col] = normalized

                # The normalized parameter values are then used to create our X and Y vectors for prediction.
                # The normalized parameter values are bucketed into `prediction_interval` sized buckets.
                # Within these buckets, the quantile values for all the specified `quantiles` functions are computed.
                # Each data point is of the form [ quantile_1, quantile_2, ..., quantile_n ].
                # This creates an initial dataset with N rows and num_quantiles columns,
                # i.e., tsdf has shape (N, num_quantiles),
                # where N is governed by the historical data available and the resampling window `prediction_interval`,
                # and num_quantiles is controlled by the `quantiles` functions used.
                quantiles = [lambda x: x.quantile(qval) for _, qval in self.quantiles_def]
                tsdf = params_df[param_col].resample(self.prediction_interval).agg(quantiles).astype(float)

                # To form the X vector, we roll consecutive `prediction_seq_len` buckets of quantile values together.
                # Each bucket from before will get the previous buckets prepended to it, padding with 0s if necessary,
                # so that each training data point has shape (seq_len, num_quantiles),
                # and ultimately X will have shape (N, seq_len, num_quantiles).
                X = [x.values for x in tsdf.rolling(window=self.prediction_seq_len)]
                for i in range(min(len(X), self.prediction_seq_len)):
                    num_pad_before = self.prediction_seq_len - X[i].shape[0]
                    pad_width = ((num_pad_before, 0), (0, 0))
                    X[i] = np.pad(X[i], pad_width)
                X = np.array(X)

                # Our Y vector is just the X-vector shifted backwards and forward-constant-filled.
                # If there isn't enough data, you may have a single NaN row,
                # in which case we fill with the last row of the original data.
                # Y has shape (N, num_quantiles).
                Y = tsdf.shift(freq=-self.prediction_horizon).reindex_like(tsdf).ffill()
                Y = Y.fillna(tsdf.iloc[-1])
                Y = Y.values

                # TODO(JACKIE): Hack to split data into two portions.
                # But we still want to train on all the data.
                if len(X) <= 1:
                    print(f"Warning: not enough data ({X.shape=}), will double up rows: {qt}.")
                    X = np.concatenate([X, X])
                    Y = np.concatenate([Y, Y])
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.1,)
                X_train, Y_train = X, Y

                # X: (N, seq_len, num_quantiles) to (seq_len, N, num_quantiles).
                X_train, X_test = np.transpose(X_train, (1, 0, 2)), np.transpose(X_test, (1, 0, 2))

                param_model = Network(
                    input_size=len(self.quantiles_def),
                    output_size=len(self.quantiles_def),
                    hidden_size=HIDDEN_SIZE,
                    num_layers=RNN_LAYERS,
                ).to(DEVICE)

                loss_function = nn.MSELoss()
                optimizer = torch.optim.Adam(param_model.parameters(), lr=LR)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(X_train.shape[1] * EPOCHS))

                epoch = None
                for epoch in range(EPOCHS):
                    train_loss = self._train_epoch(param_model, X_train, Y_train, optimizer, scheduler, loss_function)
                    val_loss = self._validate(param_model, X_test, Y_test, loss_function)
                assert epoch is not None, "No epochs?"
                # TODO(WAN): I'm not sure it makes sense to save the model for the current code.
                model[qt][param_idx]["type"] = "jackie1m1p"
                model[qt][param_idx]["jackie1m1p"] = {}
                model[qt][param_idx]["jackie1m1p"]["X"] = X
                model[qt][param_idx]["jackie1m1p"]["X_ts"] = tsdf.index
                model[qt][param_idx]["jackie1m1p"]["model"] = param_model
                model[qt][param_idx]["jackie1m1p"]["mean"] = mean
                model[qt][param_idx]["jackie1m1p"]["std"] = std
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

    ###################################################################################################
    #########################           Model Training        #########################################
    ###################################################################################################
    def _train_epoch(self, model, X_train, Y_train, optimizer, scheduler, loss_function):
        model.train()

        # Shuffle the timeseries
        arr = np.arange(X_train.shape[1])
        np.random.shuffle(arr)

        train_loss = 0
        batch_bar = tqdm(
            total=X_train.shape[1], dynamic_ncols=True, leave=False, position=0, desc="Train", disable=True
        )
        for ind in arr:
            seq = torch.tensor(X_train[:, ind : ind + 1, :]).to(DEVICE).float()
            labels = torch.tensor(Y_train[ind]).to(DEVICE).float()
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred[-1, -1, :], labels)
            single_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += float(single_loss)

            batch_bar.set_postfix(
                loss="{:.04f}".format(float(train_loss / (ind + 1))),
                lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
            )

            batch_bar.update()

        train_loss /= X_train.shape[1]
        batch_bar.close()
        return train_loss

    def _validate(self, model, X_test, Y_test, loss_function):
        # Validation loss
        model.eval()
        val_loss = 0
        batch_bar = tqdm(
            total=X_test.shape[1], dynamic_ncols=True, leave=False, position=0, desc="Validate", disable=True
        )
        for ind in range(X_test.shape[1]):
            seq = torch.tensor(X_test[:, ind : ind + 1, :]).to(DEVICE).float()
            labels = torch.tensor(Y_test[ind]).to(DEVICE).float()

            with torch.no_grad():
                y_pred = model(seq)

            single_loss = loss_function(y_pred[-1, -1, :], labels)
            val_loss += float(single_loss)
            batch_bar.update()
        val_loss /= X_test.shape[1]
        batch_bar.close()
        return val_loss

    # # Get all parameters for a query and compare it with actual data
    # def get_all_parameters_for(self, query_template: str):
    #     template_original_df = self.data_preprocessor.qt_to_original_df[query_template]
    #     template_normalized_df = self.data_preprocessor.qt_to_normalized_df[query_template]
    #     template_dtypes = self.data_preprocessor.qt_to_dtype[query_template]
    #     template_stats = self.data_preprocessor.qt_to_stats[query_template]
    #     template_index = self.qt_to_index[query_template]
    #
    #     for i, col in enumerate(template_normalized_df):
    #         # print(f"Processing parameter {i+1}...")
    #         # Skip non-numerical columns
    #         if template_dtypes[i] == "string":
    #             continue
    #
    #         # Get corresponding model
    #         model = Network(len(self.quantiles), len(self.quantiles), HIDDEN_SIZE, RNN_LAYERS)
    #         filepath = os.path.join(MODEL_SAVE_PATH, f"{template_index}_{i}")
    #         state_dict = torch.load(filepath)
    #         model.load_state_dict(state_dict["model_state"])
    #
    #         # Group by pred_iterval and get quantile data
    #         time_series_df = template_normalized_df[col].resample(self.prediction_interval).agg(self.quantiles)
    #         time_series_df = time_series_df.astype(float)
    #
    #         # Get number of queries in each time interval
    #         num_template_df = template_normalized_df[col].resample(self.prediction_interval).count()
    #         # display(num_template_df.head())
    #         # display(time_series_df.head(10))
    #
    #         # Build a new dataframe whichcontains predicted parameters for all timestamps
    #         generated_params = np.array([])
    #         num_queries_each_bin = 10
    #         timestamps = []
    #         for j in tqdm(range(len(time_series_df) - 1)):
    #             # Generate sequence data. Add padding if neccesary
    #             if j + 1 >= self.prediction_seq_len:
    #                 start_time = j - self.prediction_seq_len + 1
    #                 seq = time_series_df.iloc[start_time: (j + 1), :].to_numpy()
    #             else:
    #                 seq = time_series_df.iloc[: (j + 1), :].to_numpy()
    #                 seq = np.pad(seq, ((self.prediction_seq_len - j - 1, 0), (0, 0)))
    #
    #             # Get predicted quantiles from the model
    #             seq = seq[None, :, :]
    #             seq = np.transpose(seq, (1, 0, 2))
    #             seq = torch.tensor(seq).to(DEVICE).float()
    #             with torch.no_grad():
    #                 pred = model(seq)
    #
    #             # Ensure prediction quantile values are strictly increasing
    #             pred = pred[-1, -1, :]
    #             pred = torch.cummax(pred, dim=0).values
    #
    #             # Generate num_template samples according to the distribution defined by the predicted quantile values
    #             pred = pred.cpu().detach().numpy()
    #             # print("pred:", pred)
    #             # print("actual:", time_series_df.iloc[j+1, :].to_numpy())
    #             # Un-normalize the quantiles
    #             mean, std = template_stats[i]
    #             if std != 0:
    #                 pred = pred * std + mean
    #             else:
    #                 pred = pred + mean
    #
    #             # Draw (0.1*num_queries) samples from each bin of the predicted quantile
    #             for k in range(len(pred) - 1):
    #                 a, b = pred[k], pred[k + 1]
    #                 generated_params = np.concatenate(
    #                     [generated_params, np.random.uniform(a, b, num_queries_each_bin)]
    #                 )
    #                 for _ in range(num_queries_each_bin):
    #                     timestamps.append(num_template_df.index[j + 1])
    #
    #         # Generate a dataframe for the predicted parameter values
    #         predicted_params_df = pd.DataFrame(generated_params, index=pd.DatetimeIndex(timestamps))
    #
    #         # Graph the results
    #         min_val, max_val = template_original_df[col].min(), template_original_df[col].max()
    #         min_val = min_val - (1 + 0.2 * (max_val - min_val))
    #         max_val = max_val + (1 + 0.2 * (max_val - min_val))
    #
    #         print(f"PARAM ${i + 1} Predicted")
    #         fig, axes = joypy.joyplot(
    #             predicted_params_df.groupby(pd.Grouper(freq="5s")),
    #             hist=True,
    #             bins=20,
    #             overlap=0,
    #             grid=True,
    #             x_range=[min_val, max_val],
    #         )
    #         plt.show()
    #
    #         print(f"PARAM ${i + 1} Actual")
    #         fig, axes2 = joypy.joyplot(
    #             template_original_df[col].to_frame().groupby(pd.Grouper(freq="5s")),
    #             hist=True,
    #             bins=20,
    #             overlap=0,
    #             grid=True,
    #             x_range=[min_val, max_val],
    #         )
    #         plt.show()
    #         print("\n")
    #
    # # Get all parameters for a query and compare it with actual data
    # def get_parameters_for(self, query_template, timestamp, num_queries):
    #     target_timestamp = pd.Timestamp(timestamp)
    #     template_normalized_df = self.data_preprocessor.qt_to_normalized_df[query_template]
    #     template_dtypes = self.data_preprocessor.qt_to_dtype[query_template]
    #     template_stats = self.data_preprocessor.qt_to_stats[query_template]
    #     template_index = self.qt_to_index[query_template]
    #     param_X = self.qt_to_param_X[query_template]
    #     num_params = len(template_dtypes)
    #
    #     # Draw (0.1*num_queries) samples from each bin of the predicted quantile
    #     generated_params = np.array([])
    #     num_queries_each_bin = int(num_queries / (len(self.quantiles) - 1))
    #     num_queries_last_bin = num_queries % (len(self.quantiles) - 1)  # Left over items in last bin
    #
    #     for i in range(num_params):
    #         if template_dtypes[i] == "string":
    #             print("Skipping string columns")
    #             continue
    #
    #         # Get corresponding model
    #         model = Network(len(self.quantiles), len(self.quantiles), HIDDEN_SIZE, RNN_LAYERS)
    #         filepath = os.path.join(MODEL_SAVE_PATH, f"{template_index}_{i}")
    #         state_dict = torch.load(filepath)
    #         model.load_state_dict(state_dict["model_state"])
    #
    #         # Compute how many predictions need to be made
    #         start_timestamp = template_normalized_df.index.max()
    #         num_predictions = int((target_timestamp - start_timestamp) / self.prediction_interval)
    #
    #         # Continously make predictions until target_timestamp
    #         param_X_col = param_X[i]
    #         seq = param_X_col[-1]
    #         seq = seq[None, :, :]
    #         seq = np.transpose(seq, (1, 0, 2))
    #         seq = torch.tensor(seq).to(DEVICE).float()
    #         for j in tqdm(range(num_predictions)):
    #             # Get predicted quantiles from the model
    #             with torch.no_grad():
    #                 pred = model(seq)
    #
    #             # Ensure prediction quantile values are strictly increasing
    #             pred = pred[-1, -1, :]
    #             pred = torch.cummax(pred, dim=0).values
    #
    #             # Add pred to original seq to create new seq for next time stamp
    #             seq = torch.squeeze(seq, axis=1)
    #             seq = torch.cat((seq[1:, :], pred[None, :]), axis=0)
    #             seq = seq[:, None, :]
    #
    #         pred = pred.cpu().detach().numpy()
    #
    #         # Un-normalize the quantiles
    #         mean, std = template_stats[i]
    #         if std != 0:
    #             pred = pred * std + mean
    #         else:
    #             pred = pred + mean
    #
    #         # Draw (0.1*num_queries) samples from each bin of the predicted quantile
    #         for k in range(len(pred) - 1):
    #             a, b = pred[k], pred[k + 1]
    #             if j == len(pred) - 2:
    #                 generated_params = np.concatenate(
    #                     [generated_params, np.random.uniform(a, b, num_queries_last_bin)]
    #                 )
    #             else:
    #                 generated_params = np.concatenate(
    #                     [generated_params, np.random.uniform(a, b, num_queries_each_bin)]
    #                 )
    #     return generated_params


if __name__ == "__main__":
    fmd = ForecastMD.load("fmd.pkl")

    # Fit the forecaster object.
    forecaster = Jackie1m1p()
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
