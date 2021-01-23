import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.expand(timestep, -1, -1).transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return attn_energies.softmax(2)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.expand(encoder_outputs.size(0), -1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id, n_layers=1):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, n_layers)

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, input_, last_hidden, encoder_outputs):
        emb = self.emb(input_.transpose(0, 1))
        attn = self.attention(last_hidden, encoder_outputs)
        context = attn.bmm(encoder_outputs).transpose(0, 1)
        rnn_input = torch.cat((emb, context), dim=2)

        outputs, hidden = self.rnn(rnn_input, last_hidden)

        if outputs.requires_grad:
            outputs.register_hook(lambda x: x.clamp(min=-10, max=10))

        outputs = self.out(outputs.contiguous().squeeze(0)).log_softmax(1)

        return outputs, hidden

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                teacher_forcing_ratio=0):
        inputs, batch_size, max_length = self._validate_args(
            inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio)

        use_teacher_forcing = True if torch.rand(
            1).item() < teacher_forcing_ratio else False

        outputs = []

        self.rnn.flatten_parameters()

        decoder_hidden = torch.zeros(
            1, batch_size, self.hidden_size, device=encoder_outputs.device)

        def decode(step_output):
            symbols = step_output.topk(1)[1]
            return symbols

        if use_teacher_forcing:
            for di in range(max_length+1):
                decoder_input = inputs[:, di].unsqueeze(1)

                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs)

                step_output = decoder_output.squeeze(1)
                outputs.append(step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                step_output = decoder_output.squeeze(1)
                outputs.append(step_output)

                symbols = decode(step_output)
                decoder_input = symbols

        outputs = torch.stack(outputs).permute(1, 0, 2)

        return outputs, decoder_hidden

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        batch_size = encoder_outputs.size(0)

        if inputs is None:
            assert teacher_forcing_ratio == 0

            inputs = torch.full((batch_size, 1), self.sos_id,
                                dtype=torch.long, device=encoder_outputs.device)

            max_length = self.max_len
        else:
            max_length = inputs.size(1) - 1

        return inputs, batch_size, max_length
