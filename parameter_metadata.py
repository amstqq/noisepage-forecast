import pickle
from multiprocessing import cpu_count

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from ddsketch import DDSketch
from pandas.api.types import is_datetime64_any_dtype
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class ForecastMD:
    SESSION_BEGIN = "SESSION_BEGIN"
    SESSION_END = "SESSION_END"

    def __init__(self):
        self.qtmds: Dict[QueryTemplateMD] = {}
        self.qt_enc = QueryTemplateEncoder()
        # Dummy tokens for session begin and session end.
        self.qt_enc.fit([self.SESSION_BEGIN, self.SESSION_END, pd.NA])

        # networkx dict_of_dicts format.
        self.transition_sessions = {}
        self.transition_txns = {}

        self.arrivals = []

        self.cache = {}

    def augment(self, df):
        # Invalidate the cache.
        self.cache = {}

        # Sort the dataframe. All the below code assumes that the dataframe is sorted chronologically.
        df = df.sort_values(["log_time", "session_line_num"])

        # Drop any torn txns from the df.
        # Specifically, it is assumed that each txn has a clear begin and end marker.
        valid_starts = ["BEGIN"]
        valid_ends = ["COMMIT", "ROLLBACK"]
        vtxid_group = df.groupby("virtual_transaction_id", sort=False)
        good_starts = vtxid_group.nth(0)["query_template"].isin(valid_starts)
        good_starts = good_starts[good_starts].index
        good_ends = vtxid_group.nth(-1)["query_template"].isin(valid_ends)
        good_ends = good_ends[good_ends].index
        good_vtxids = (good_starts.intersection(good_ends)).values

        pre_drop_rows = df.shape[0]
        df = df.drop(df[~df["virtual_transaction_id"].isin(good_vtxids)].index)
        post_drop_rows = df.shape[0]
        dropped_rows = pre_drop_rows - post_drop_rows
        if dropped_rows > 0:
            print(
                f"Dropped {dropped_rows} rows belonging to torn/unconforming transactions. {post_drop_rows} rows remain."
            )

        # Augment the dataframe while updating internal state.

        # Encode the query templates.
        print("Encoding query templates.")
        df["query_template_enc"] = self.qt_enc.fit_transform(df["query_template"])

        # Lagged time.
        df["think_time"] = (df["log_time"] - df["log_time"].shift(1)).shift(-1).dt.total_seconds()

        def record(row):
            qt_enc = row["query_template_enc"]
            if qt_enc not in self.qtmds:
                qt = row["query_template"]
                self.qtmds[qt_enc] = QueryTemplateMD(
                    query_template=qt, query_template_encoding=self.qt_enc.transform(qt)
                )
            self.qtmds[qt_enc].record(row)

        print("Recording query template info.")
        df.apply(record, axis=1)

        print("Updating transitions for sessions.")
        self._update_transition_dict(self.transition_sessions, self._compute_transition_dict(df, "session_id"))
        print("Updating transitions for transactions.")
        self._update_transition_dict(self.transition_txns, self._compute_transition_dict(df, "virtual_transaction_id"))

        # We need to keep the arrivals around.
        # Assumption: every transaction starts with a BEGIN.
        # Therefore, only the BEGIN entries need to be considered.
        # TODO(WAN): Other ways of starting transactions.
        print("Keeping historical arrival times.")
        begin_times = df.loc[df["query_template"] == "BEGIN", "log_time"]
        self.arrivals.append(begin_times)

    def visualize(self, target):
        assert target in ["sessions", "txns"], f"Bad target: {target}"

        if target == "sessions":
            transitions = self.transition_sessions
        else:
            assert target == "txns"
            transitions = self.transition_txns

        def rewrite(s):
            l = 24
            return "\n".join(s[i : i + l] for i in range(0, len(s), l))

        G = nx.DiGraph(transitions)
        nx.relabel_nodes(G, {k: rewrite(self.qt_enc.inverse_transform(k)) for k in G.nodes}, copy=False)
        AG = nx.drawing.nx_agraph.to_agraph(G)
        AG.layout("dot")
        AG.draw(f"{target}.pdf")

    @staticmethod
    def _update_transition_dict(current, other):
        for src in other:
            current[src] = current.get(src, {})
            for dst in other[src]:
                current[src][dst] = current[src].get(dst, {"weight": 0})
                current[src][dst]["weight"] += other[src][dst]["weight"]
                # Set the label for printing.
                current[src][dst]["label"] = current[src][dst]["weight"]

    def _compute_transition_dict(self, df, group_key):
        assert group_key in ["session_id", "virtual_transaction_id"], f"Unknown group key: {group_key}"

        group_fn = None
        if group_key == "session_id":
            group_fn = self._group_session
        elif group_key == "virtual_transaction_id":
            group_fn = self._group_txn
        assert group_fn is not None, "Forgot to add a case?"

        transitions = {}
        groups = df.groupby(group_key)
        chunksize = max(1, len(groups) // cpu_count())
        grouped = process_map(group_fn, groups, chunksize=chunksize, desc=f"Grouping on {group_key}.", disable=True)
        # TODO(WAN): Parallelize.
        for group_id, group_qt_encs in tqdm(
            grouped, desc=f"Computing transition matrix for {group_key}.", disable=True
        ):
            for transition in zip(group_qt_encs, group_qt_encs[1:]):
                src, dst = transition
                transitions[src] = transitions.get(src, {})
                transitions[src][dst] = transitions[src].get(dst, {"weight": 0})
                transitions[src][dst]["weight"] += 1
                transitions[src][dst]["label"] = transitions[src][dst]["weight"]
        return transitions

    def _group_txn(self, item):
        group_id, df = item
        qt_encs = df["query_template_enc"].values
        return group_id, qt_encs

    def _group_session(self, item):
        group_id, df = item
        qt_encs = df["query_template_enc"].values
        qt_encs = np.concatenate(
            [self.qt_enc.transform([self.SESSION_BEGIN]), qt_encs, self.qt_enc.transform([self.SESSION_END]),]
        )
        return group_id, qt_encs

    def get_qtmd(self, query_template):
        """
        Parameters
        ----------
        query_template : str

        Returns
        -------
        qtmd : QueryTemplateMD
        """
        encoding = self.qt_enc.transform(query_template)
        return self.qtmds[encoding]

    def get_cache(self):
        return self.cache

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
