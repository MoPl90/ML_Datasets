import torch


class RNN(torch.nn.Module):

    name = "RNN"

    def __init__(
        self,
        embedding_dim,
        vocab_size,
        rnn_hidden_dim,
        hidden_dim,
        dropout_p,
        out_dim=1,
        padding_idx=-1,
        **kwargs
    ):
        super().__init__()

        # Initialize embeddings
        self.embeddings = torch.nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            padding_idx=padding_idx,
        )

        # RNN
        self.rnn = torch.nn.GRU(
            embedding_dim,
            rnn_hidden_dim,
            batch_first=True,
            bidirectional=True,  # There is no natural/causal order of the sequence!
        )

        # FC weights
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(
            rnn_hidden_dim * 2, hidden_dim
        )  # factor of 2 for bidirectional!
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """RNN forward pass

        Args:
            x (torch.tensor): input tensor

        Returns:
            torch.tensor: output tensor
        """
        # Embed
        x_in = x
        x_in = self.embeddings(x_in)

        # RNN outputs
        out, h_n = self.rnn(x_in)
        z = out[:, -1]  # assume that the batch has been padded

        # Prediction head
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return z
