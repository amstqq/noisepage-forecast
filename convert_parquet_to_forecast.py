import math
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from tqdm import tqdm, trange

from constants import (
    DEBUG_FORECAST_FOLDER,
    DEBUG_POSTGRESQL_PARQUET_TRAIN,
    DEBUG_POSTGRESQL_PARQUET_FUTURE,
    PG_LOG_DTYPES,
    TXN_AWARE_PARAM_FORECAST,
)
from forecast_metadata import ForecastMD
from generated_forecast_md import GeneratedForecastMD


def generate_forecast_arrivals(fmd, target_timestamp, granularity, plot):
    """

    Parameters
    ----------
    fmd : ForecastMD
        The forecast metadata that has the arrival times stored.

    target_timestamp : str
        The timestamp into the future that should be forecasted until.

    granularity : pd.Timedelta
        The granularity at which forecasting should be performed.
        For example, pd.Timedelta(hours=1) will forecast data at an hour granularity.

    plot : bool
        If true, will draw plots.

    Returns
    -------
    forecasted_arrivals : pd.DataFrame
    """

    # Assumption: every transaction starts with a BEGIN.
    # Therefore, only the BEGIN entries need to be considered.
    # TODO(WAN): Other ways of starting transactions.
    arrivals = pd.concat(fmd.arrivals).to_frame(name="log_time")
    begin_times = arrivals.set_index("log_time").resample(granularity).size()

    # TODO(WAN): Make timezone code more robust.
    begin_times = begin_times.tz_convert("UTC").tz_convert(None)

    # Convert the dataframe to NeuralProphet format.
    ndf = begin_times.iloc[:-1].to_frame()
    ndf = ndf.reset_index().rename(columns={"log_time": "ds", 0: "y"})

    # Get the forecast time range.
    ts_last = ndf["ds"].max()
    horizon_target = pd.to_datetime(target_timestamp).tz_convert("UTC").tz_convert(None)
    assert horizon_target > ts_last, "Horizon is not in the future?"

    # ndf_forecast is the forecast dataframe that needs to have its "y" columns filled in.
    # NeuralProphet's make_future_dataframe is not used because it doesn't seem to expose the freq.
    dr = pd.date_range(start=ts_last, end=horizon_target, freq=granularity, inclusive="right")
    ndf_forecast = pd.DataFrame({"ds": dr, "y": [None] * len(dr)})
    ndf_forecast = pd.concat([ndf, ndf_forecast])

    model = NeuralProphet()
    # Train NeuralProphet on the original data.
    metrics = model.fit(ndf, freq=granularity)
    # Forecast the future.
    forecasted_arrivals = model.predict(ndf_forecast)

    if plot:
        Path("./artifacts/plots/").mkdir(parents=True, exist_ok=True)
        model.plot(forecasted_arrivals, xlabel="Timestamp", ylabel="Arrivals")
        plt.savefig("./artifacts/plots/forecasted_arrivals.pdf")
        model.plot_components(forecasted_arrivals)
        plt.savefig("./artifacts/plots/forecasted_arrivals_components.pdf")

    forecasted_arrivals.ds = forecasted_arrivals.ds.dt.tz_localize("UTC")
    return forecasted_arrivals


def generate_forecast(fmd, target_timestamp, granularity=pd.Timedelta(hours=1), plot=False):
    """

    Parameters
    ----------
    fmd : ForecastMD
    target_timestamp : str
    granularity : pd.Timedelta
    plot : bool

    Returns
    -------

    """
    out_folder = Path(DEBUG_FORECAST_FOLDER)
    shutil.rmtree(out_folder, ignore_errors=True)
    out_folder.mkdir(parents=True, exist_ok=True)

    with open(out_folder / f"generated_forecast_md.pkl", "wb") as f:
        metadata = GeneratedForecastMD(target_timestamp, granularity)
        pickle.dump(metadata, f)

    forecast_arrivals = generate_forecast_arrivals(fmd, target_timestamp, granularity, plot)
    model = fmd.get_cache()["forecast_model"]["jackie1m1p"]
    # model = fmd.get_cache()["forecast_model"]["distfit"]

    for i, row in tqdm(
        forecast_arrivals.iterrows(),
        total=forecast_arrivals.shape[0],
        desc="Generating sessions until the forecast horizon.",
    ):
        current_ts = row.ds
        num_forecasted_sessions = math.ceil(row.yhat1)
        destination_parquet = out_folder / f"{current_ts}.parquet"

        # Generate a sample path for each session.
        rows = []
        for session_num in trange(num_forecasted_sessions, desc=f"Generating sessions for {current_ts}.", leave=False):
            current_session_ts = current_ts
            session_line_num = 1

            qte_cur = fmd.qt_enc.transform("BEGIN")
            qte_ends = fmd.qt_enc.transform(["COMMIT", "ROLLBACK"])
            qt_cur = fmd.qt_enc.inverse_transform(qte_cur)
            params_cur = {}

            # Generate a sample path for the current session.
            sample_path = []
            first_qte = None
            while True:
                # Emit.
                sample_path.append((session_num, session_line_num, qt_cur, params_cur, qte_cur))
                # Stop.
                if qte_cur in qte_ends:
                    break

                # Advance the session line number.
                session_line_num += 1

                # Pick the next query template by sampling the Markov chain.
                transitions = fmd.transition_txns[qte_cur].items()
                candidate_templates = [k for k, _ in transitions]
                probs = np.array([v["weight"] for _, v in transitions])
                probs = probs / np.sum(probs)
                qte_cur = np.random.choice(candidate_templates, p=probs)

                # The first query of a sample path (other than BEGIN) is used to identify
                # a transaction.
                if first_qte == None:
                    first_qte = qte_cur

                # Generate the parameters.
                qt_cur = fmd.qt_enc.inverse_transform(qte_cur)
                if TXN_AWARE_PARAM_FORECAST:
                    params_cur = model.generate_parameters_txn_aware(
                        qt_cur,
                        qte_cur,
                        current_session_ts,
                        fmd.txn_to_transition_params[first_qte],
                        fmd.get_cache()["db_schema"],
                        sample_path,
                    )
                else:
                    params_cur = model.generate_parameters(qt_cur, current_session_ts)

                # Advance the time.
                current_session_ts += pd.Timedelta(
                    seconds=fmd.qtmds[qte_cur]._think_time_sketch.get_quantile_value(0.5)
                )
            # Write the sample path.
            for session_num, session_line_num, qt, params, _ in sample_path:
                rows.append([session_num, session_line_num, qt, params])
                # rows.append([session_num, session_line_num, qt, tuple(v for _, v in sorted(params.items()))])

        dtypes = {
            "session_id": str,
            "session_line_num": "Int64",
            "query_template": str,
            "query_params": object,
        }
        df = pd.DataFrame(rows, columns=dtypes.keys())
        df = df.astype(dtype=dtypes)
        df.to_parquet(destination_parquet)
        # print("EARLY STOPPING FOR DEBUG.")
        # break


def main():
    # fmd = ForecastMD()
    # pq_files = [Path(DEBUG_POSTGRESQL_PARQUET_TRAIN)]
    # print(f"Parquet files: {pq_files}")
    # for pq_file in tqdm(pq_files, desc="Reading Parquet files.", disable=True):
    #     df = pd.read_parquet(pq_file)
    #     df["log_time"] = df["log_time"].dt.tz_convert("UTC")
    #     print(f"{pq_file} has timestamps from {df['log_time'].min()} to {df['log_time'].max()}.")
    #     df["query_template"] = df["query_template"].replace("", np.nan)
    #     dropna_before = df.shape[0]
    #     df = df.dropna(subset=["query_template"])
    #     dropna_after = df.shape[0]
    #     print(
    #         f"Dropped {dropna_before - dropna_after} empty query template rows in {pq_file}. {dropna_after} rows remain."
    #     )
    #     fmd.augment(df)
    # fmd.save("fmd.pkl")

    fmd = ForecastMD.load("fmd.pkl")

    # Precompute a bunch of stuff.
    cache = fmd.get_cache()
    if "forecast_model" not in cache:
        cache["forecast_model"] = {}
        fmd.save("fmd.pkl")

    # distfit model.
    if "distfit" not in cache["forecast_model"]:
        from fm_distfit import DistfitModel

        cache["forecast_model"]["distfit"] = DistfitModel().fit(fmd)
        fmd.save("fmd.pkl")

    # Jackie's 1m1p model.
    if "jackie1m1p" not in cache["forecast_model"]:
        from fm_jackie import Jackie1m1p

        cache["forecast_model"]["jackie1m1p"] = Jackie1m1p().fit(fmd)
        fmd.save("fmd.pkl")

    # database schema
    # if "db_schema" not in cache:
    from forecast_schema import get_database_schema

    cache["db_schema"] = get_database_schema()
    fmd.save("fmd.pkl")

    fmd = ForecastMD.load("fmd.pkl")

    arrivals = pd.concat(fmd.arrivals).dt.tz_convert("UTC")

    # TODO(WAN): This can be cached for performance.
    future_df = pd.read_parquet(DEBUG_POSTGRESQL_PARQUET_FUTURE)
    forecast_target = future_df["log_time"].max().tz_convert("UTC")

    print(f"Generating forecast. [{arrivals.min()}, {arrivals.max()}] to {forecast_target}.")
    generate_forecast(fmd, forecast_target, pd.Timedelta(seconds=1), plot=False)


if __name__ == "__main__":
    main()
