import torch.nn as nn
import torch


class DeepAR(nn.Module):
    def __init__(
        self, quantile_dim, cov_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, embedding_dim, num_classes, device
    ):
        """
        DeepAR model that predicts future values of a time-dependent variable
        based on past values and covariates.
        """
        super().__init__()

        self.quantile_dim = quantile_dim
        self.cov_dim = cov_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.device = device

        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=quantile_dim + cov_dim + embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bias=True,
            batch_first=False,
            dropout=lstm_dropout,
        )

        self.relu = nn.ReLU()

        linear_dim = lstm_hidden_dim * lstm_layers

        self.quantiles = nn.Linear(linear_dim, quantile_dim)
        self._init_weights()

    def _init_weights(self):
        # initialize LSTM forget gate bias to 1 as recommanded by
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

    def forward(self, zxs, class_ids, hidden, cell):
        """
        zxs ([1, B, quantile_dim+cov_dim])
        class_ids ([1, B])
        """
        idx_embed = self.embedding(class_ids)  # (1, B, embed_dim)
        lstm_input = torch.cat((zxs, idx_embed), dim=2)  # (1, B, q_dim+cov_dim+embed_dim)

        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # use hidden from all lstm layers to compute quantiles
        # hidden_permute: (num_layers, B, hidden_dim) -> (B, hidden_dim, num_layers) -> (B, hidden_dim * num_layers)
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)

        output = self.quantiles(hidden_permute)  # (B, quantile_dim)
        return output, hidden, cell

    def init_hidden(self, input_size):
        """
        Initialize hidden states
        """
        return torch.zeros(self.lstm_layers, input_size, self.lstm_hidden_dim, device=self.device)

    def init_cell(self, input_size):
        """
        Initialize cell states
        """
        return torch.zeros(self.lstm_layers, input_size, self.lstm_hidden_dim, device=self.device)

    def test(self, zxs, class_ids, hidden, cell, history_window_size, prediction_window_size):
        # zxs: (T, B, q_dim + cov_dim)
        # scaling_factors: (N, q_dim + cov_dim)
        B = zxs.shape[1]
        pred_quantiles = torch.zeros((prediction_window_size, B, self.quantile_dim), device=self.device)
        h, c = hidden, cell
        for t in range(prediction_window_size):
            quantiles, h, c = self(zxs[history_window_size + t].unsqueeze(0), class_ids, h, c)  # (B, q_dim)

            pred_quantiles[t] = quantiles

            # Replace zxs at next timestamp by the predicted quantile values
            if t < (prediction_window_size - 1):
                zxs[history_window_size + t + 1, :, : self.quantile_dim] = quantiles

        return pred_quantiles
