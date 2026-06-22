import torch
import torch.nn as nn


class ManualLSTMCell(nn.Module):
    """
    手写单步 LSTM cell
    输入:
        x_t: (B, input_dim)
        h_prev: (B, hidden_dim)
        c_prev: (B, hidden_dim)

    输出:
        h_t: (B, hidden_dim)
        c_t: (B, hidden_dim)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_x = nn.Linear(input_dim, 4 * hidden_dim)
        self.W_h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x_t, h_prev, c_prev):
        gates = self.W_x(x_t) + self.W_h(h_prev)

        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)

        i_t = torch.sigmoid(i_t)   # input gate
        f_t = torch.sigmoid(f_t)   # forget gate
        g_t = torch.tanh(g_t)      # candidate memory
        o_t = torch.sigmoid(o_t)   # output gate

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class LSTM(nn.Module):
    """
    手写版多层 LSTM

    输入:
        x: (B, T, input_dim)

    输出:
        output: (B, T, hidden_dim)
        h_n:    (num_layers, B, hidden_dim)
        c_n:    (num_layers, B, hidden_dim)

    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
        batch_first=True,
    ):
        super().__init__()

        assert batch_first is True, "这里只实现 batch_first=True 的情况"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            self.layers.append(
                ManualLSTMCell(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                )
            )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        x: (B, T, input_dim)
        """

        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if hidden is None:
            h = [
                torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
            c = [
                torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_0, c_0 = hidden
            h = [h_0[layer_idx] for layer_idx in range(self.num_layers)]
            c = [c_0[layer_idx] for layer_idx in range(self.num_layers)]

        layer_input = x

        h_n = []
        c_n = []

        for layer_idx, lstm_cell in enumerate(self.layers):
            outputs = []

            h_t = h[layer_idx]
            c_t = c[layer_idx]

            for t in range(T):
                x_t = layer_input[:, t, :]

                h_t, c_t = lstm_cell(x_t, h_t, c_t)

                outputs.append(h_t.unsqueeze(1))

            layer_output = torch.cat(outputs, dim=1)

            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)

            layer_input = layer_output

            h_n.append(h_t.unsqueeze(0))
            c_n.append(c_t.unsqueeze(0))

        output = layer_input
        h_n = torch.cat(h_n, dim=0)
        c_n = torch.cat(c_n, dim=0)

        return output, (h_n, c_n)



if __name__ == "__main__":
    B = 8
    T = 64
    input_dim = 128
    hidden_dim = 256
    num_layers = 2

    x = torch.randn(B, T, input_dim)

    lstm = LSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        batch_first=True,
    )

    output, (h_n, c_n) = lstm(x)

    print("Output shape:", output.shape)  # (B, T, hidden_dim)
    print("h_n shape:", h_n.shape)        # (num_layers, B, hidden_dim)
    print("c_n shape:", c_n.shape)        # (num_layers, B, hidden_dim)

