from forecast_models import ForecastModelABC
from distfit import distfit
from tqdm import tqdm
import numpy as np

from collections import defaultdict
from constants import TXN_AWARE_PARAM_NEW_VAL_TOKEN


class DistfitModel(ForecastModelABC):
    def fit(self, forecast_md):
        model = {}
        for qt_enc, qtmd in tqdm(
            forecast_md.qtmds.items(), total=len(forecast_md.qtmds), desc="Fitting query templates."
        ):
            qt = forecast_md.qt_enc.inverse_transform(qt_enc)
            print(f"Fitting query template {qt_enc}: {qt}")
            model[qt] = {}
            params = qtmd.get_historical_params()

            if len(params) == 0:
                # No parameters.
                continue

            for idx, col in enumerate(params.columns, 1):
                model[qt][idx] = {}
                if str(params[col].dtype) == "string":
                    model[qt][idx]["type"] = "sample"
                    model[qt][idx]["sample"] = params[col]
                    print(f"Query template {qt_enc} parameter {idx} is a string. " "Storing values to be sampled.")
                else:
                    assert not str(params[col].dtype) == "object", "Bad dtype?"
                    dist = distfit()
                    dist.fit_transform(params[col], verbose=0)
                    print(
                        f"Query template {qt_enc} parameter {idx} "
                        f"fitted to distribution: {dist.model['distr'].name} {dist.model['params']}"
                    )
                    model[qt][idx]["type"] = "distfit"
                    model[qt][idx]["distfit"] = dist
        self.model = model
        return self

    def generate_parameters(self, query_template, timestamp):
        # The timestamp is unused because we are just fitting a distribution.
        # Generate the parameters.
        params = {}
        for param_idx in self.model[query_template]:
            fit_obj = self.model[query_template][param_idx]
            fit_type = fit_obj["type"]

            param_val = None
            if fit_type == "distfit":
                dist = fit_obj["distfit"]
                param_val = str(dist.generate(n=1, verbose=0)[0])
            else:
                assert fit_type == "sample"
                param_val = np.random.choice(fit_obj["sample"])
            assert param_val is not None

            # Param dict values must be quoted for consistency.
            params[f"${param_idx}"] = f"'{param_val}'"
        return params

    def generate_parameters_txn_aware(
        self, query_template, query_template_encoding, timestamp, transition_params, sample_path
    ):
        def sample_new_param(query_template, param_idx):
            fit_obj = self.model[query_template][param_idx]
            fit_type = fit_obj["type"]

            param_val = None
            if fit_type == "distfit":
                dist = fit_obj["distfit"]
                param_val = str(dist.generate(n=1, verbose=0)[0])
            else:
                assert fit_type == "sample"
                param_val = np.random.choice(fit_obj["sample"])
            assert param_val is not None
            return param_val

        # Dict that maps how many times each query template appears in sample_path. This is used
        # to filter the parameter transition dict, since a parameter might depend on many different
        # params that appear before it
        qt_encs_count = defaultdict(int)
        for _, _, _, _, qt_enc in sample_path:
            qt_encs_count[qt_enc] += 1

        params = {}
        for param_idx in self.model[query_template]:
            # Extract dependencies of current parameters wrt. all parameters seen in sample_path
            qtp_enc = f"{query_template_encoding}_{param_idx}"

            # Current parameter does not exist in transition dict
            if qtp_enc not in transition_params:
                param_val = sample_new_param(query_template, param_idx)
                # Param dict values must be quoted for consistency.
                params[f"${param_idx}"] = f"'{param_val}'"
                continue

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
                param_val = sample_new_param(query_template, param_idx)
                params[f"${param_idx}"] = f"'{param_val}'"
                continue

            final_probs = np.array(final_probs)
            final_probs = final_probs / np.sum(final_probs)
            qtp_enc_most_recent = np.random.choice(final_candidates, p=final_probs)

            # Selected transition is NEW_VAL_TOKEN, meaning a new value is to be sampled
            if qtp_enc_most_recent == TXN_AWARE_PARAM_NEW_VAL_TOKEN:
                param_val = sample_new_param(query_template, param_idx)
                params[f"${param_idx}"] = f"'{param_val}'"
                continue

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

            # Previous param values have been quoted. No need to quote again.
            params[f"${param_idx}"] = param_val
        return params
