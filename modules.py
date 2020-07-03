import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP with one hidden layer
    """
    def __init__(self, in_dim, h_dim, out_dim):
        super(MLP, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        y = self.deterministic_output(x)
        return y

class SimpleAttention(nn.Module):
    """
    Simple attention mechanism used in DialogueRNN (see GitHub page)
    Addition: selective attention according to mask
    """

    def __init__(self, dim):
        """
        :param dim: dimension of input data
        """
        super(SimpleAttention, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, 1, bias=False)

    def forward(self, C, x=None, mask=None):
        """
        :param C: (seq_len, batch, dim)
        :param x: Dummy argument for compatibility with MatchingAttention
        :param mask: (seq_len, batch). Array of boolean values indicating what a token should be considered to compute attention (True)
        """
        Z = self.linear(C).permute(1, 2, 0) # (batch, 1, seq_len)

        # Select sequence elements to which apply attention
        #
        # Note that this functionality only works if, at least, one element of the input sequence
        # is selected to compute attention scores
        #
        # TODO: Compute attention scores for selected elements only
        #  while avoiding the case where no element is selected
        #  (for us, this corresponds to the case where the robot hasn't spoken yet)

        # if mask:  # Replace tokens that do not participate in attention computation with -inf
        #     mask_ = mask.permute(1, 0).unsqueeze(1)  # mask_: (batch, 1, seq_len)
        #     Z[~mask_] = -float('inf')

        alpha = F.softmax(Z, dim=2) # (batch, 1, seq_len)
        attention_pool = torch.bmm(alpha, C.transpose(0, 1))[:, 0, :] # (batch, dim)

        # TODO: Return attention weights for logging
        return attention_pool # , alpha[:, 0, :] # alpha: (batch, seq_len)

class MatchingAttention(nn.Module):
    """
    General matching attention mechanism used in DialogueRNN (see GitHub page)
    Addition: selective attention according to mask
    """

    def __init__(self, input_dim, mem_dim):
        """

        :param input_dim: dimension of data vector
        :param mem_dim: dimension of projection "memory" space
        """
        super(MatchingAttention, self).__init__()
        self.input_dim = input_dim
        self.mem_dim = mem_dim
        self.linear = nn.Linear(self.input_dim, self.mem_dim, bias=False)

    def forward(self, C, x, mask=None):
        """
        :param C: (seq_len, batch, mem_dim)
        :param x: (batch, input_dim)
        :param mask: (seq_len, batch). Array of boolean values indicating what a token should be considered to compute attention (True)
        :return: attention_pool: (batch, mem_dim)
        """
        C_ = C.permute(1, 2, 0) # C_: (batch, mem_dim, seq_len)
        x_ = x.unsqueeze(1) # x_: (batch, 1, input_dim)

        Z = torch.bmm(self.linear(x_), C_) # (batch, 1, seq_len)

        # Select sequence elements to which apply attention
        #
        # Note that this functionality only works if at least one element of the input sequence
        # is selected to compute attention scores
        #
        # TODO: Compute attention scores for selected elements only
        #  while avoiding the case where no element is selected
        #  (for us, this corresponds to the case where the robot hasn't spoken yet)

        # if mask:  # Replace tokens that do not participate in attention computation with -inf
        #     mask_ = mask.permute(1, 0).unsqueeze(1)  # mask_: (batch, 1, seq_len)
        #     Z[~mask_] = -float('inf')

        alpha = F.softmax(Z, dim=2) # (batch, 1, seq_len)
        attention_pool = torch.bmm(alpha, C_.transpose(1, 2))[:, 0, :] # (batch, mem_dim)

        # TODO: Return attention weights for logging
        return attention_pool # , alpha[:, 0, :] # alpha: (batch, seq_len)

class SimpleRNN(nn.Module):
    """
    A simple GRU with attention (when the option is on). Used as a baseline to compare HRI-RNN against
    """

    def __init__(self, input_dim, hidden_dim, user_data_only=False, attend_over_context=0):
        super(SimpleRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.user_data_only = user_data_only
        self.attend_over_context = attend_over_context

        if attend_over_context == 1: # For now, I only use SimpleAttention for the baseline
            self.attention = SimpleAttention(hidden_dim)

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1)

    def forward(self, data):
        """
        :param data: (seq_len, batch_size, input_dim)
        :return: res: batch_size, hidden_dim)
                 If attention is on, a weighted sum of all hidden states.
                 Else, returns the last hidden state of the GRU.
        """
        if self.user_data_only:
            # Mask robot audio in input data
            for i in range(data.shape[0]): # data is of shape (seq_length, batch_size, nb_features)
                for j in range(data.shape[1]):
                    if data[i][j][-1] > 0: # User is listening to robot
                        data[i][j][32:64] = torch.zeros(64 - 32) # Mask robot audio (audio features are between indices 32--63)

        output, h = self.rnn(data)
        # output: (seq_len, batch_size, hidden_dim). All intermediate hidden states
        # h: (1, batch_size, hidden_dim)

        if self.attend_over_context == 1:
            # We apply attention over all intermediate hidden states
            res = self.attention(output)
        else: # Any value other than 1 is considered as "no attention"
            # We return the last hidden state of the GRU's last layer
            h = h.permute(1, 0, 2).mean(dim=1)  # h: (batch_size, 1, hidden_dim)
            res = h.view(-1, self.hidden_dim)  # res: (batch_size, hidden_dim)

        return res

class HriRNNCell(nn.Module):
    def __init__(self, input_dim, u_dim, c_dim, attend_over_context=0):
        """
        :param input_dim: dimension of HRI data (number of extracted audio & video features)
        :param u_dim: dimension of hidden state of user GRU cell
        :param c_dim: dimension of hidden state of context GRU cell
        :param attend_over_context: default (0): no attention, 1: SimpleAttention, 2: MatchingAttention
        """
        super(HriRNNCell, self).__init__()

        self.input_dim = input_dim
        self.u_dim = u_dim
        self.c_dim = c_dim

        self.context_cell = nn.GRUCell(32 + u_dim, c_dim) # 32: number of audio features
        self.user_state_cell = nn.GRUCell(input_dim + c_dim, u_dim)

        self.attend_over_context = attend_over_context

        if attend_over_context == 1:
            self.attention = SimpleAttention(self.c_dim)
        elif attend_over_context == 2:
            self.attention = MatchingAttention(self.input_dim, self.c_dim)

    def forward(self, data, u_prev, c_hist):
        # data: (batch, input_dim)
        # u_prev: (batch, u_dim)
        # c_hist: (t - 1, batch, c_dim)

        # TODO:
        #  - Check initializations works fine when on GPU
        #  - Return attention weights
        #  - Add default case where the context is automatically updated at each time step

        # Select data where the robot is speaking (from input batch)
        # This information is given by the last feature (< 0: user speaking, > 0: robot speaking)
        condition = data[:, -1] > 0
        indices_robot_is_speaking = torch.nonzero(condition, as_tuple=True)[0]

        # Prepare user data for user state GRU cell. Audio data (features 32 to 63) is replaced with zeros if it belongs to the robot
        data_u = data.clone()
        data_u[indices_robot_is_speaking][:, 32:64] = torch.zeros(indices_robot_is_speaking.size()[0], 64 - 32)

        # Create and initialize context vectors tensor. Size: (batch, c_dim)
        c = torch.zeros(data.size()[0], self.c_dim).double() if c_hist.size()[0] == 0 else c_hist[-1]

        # Update context only if the robot is speaking. Otherwise, we keep past the previous context
        c[indices_robot_is_speaking] = self.context_cell(torch.cat((data[indices_robot_is_speaking][:, 32:64], u_prev[indices_robot_is_speaking]), dim=1),
                                                         torch.zeros(indices_robot_is_speaking.size()[0], self.c_dim).double() if c_hist.size()[0] == 0 else
                                                         c_hist[-1][indices_robot_is_speaking])

        if self.attend_over_context:
            if c_hist.size()[0] == 0:
                input_c = torch.zeros(data.size()[0], self.c_dim).type(data.type()) # (batch, c_dim)
            else:
                input_c = self.attention(c_hist, data_u)
        else:
            input_c = c
        u = self.user_state_cell(torch.cat((data_u, input_c), dim=1), u_prev)

        return u, c

class HriRNN(nn.Module):
    def __init__(self, input_dim, u_dim, c_dim, use_gpu=False, attend_over_context=0):
        """
        :param input_dim: dimension of HRI data (number of extracted audio & video features)
        :param u_dim: dimension of hidden state of user GRU cell
        :param c_dim: dimension of hidden state of context GRU cell
        :param use_gpu: use GPU if True and CPU otherwise
        :param attend_over_context: default (0): no attention, 1: SimpleAttention, 2: MatchingAttention
        """
        super(HriRNN, self).__init__()
        self.input_dim = input_dim  # Dimension of user data
        self.u_dim = u_dim  # Dimension of hidden state of user GRU cell
        self.c_dim = c_dim  # Dimension of hidden state of context GRU cell

        self.hri_cell = HriRNNCell(input_dim, u_dim, c_dim, attend_over_context)
        self.hri_cell.double()

        self.use_gpu = use_gpu

    def forward(self, sequence):
        """
        :param sequence: Sequence of data frames of size (seq_len, batch, input_dim)
        :return: ut: Last user state in the sequence of size (batch, u_dim)
        """

        # TODO:
        #  - Return attention weights

        # Initialize hidden states for context & user GRU cells (t = 0)
        if self.use_gpu:
            ut = torch.zeros(sequence.size()[1], self.u_dim).cuda().double() # size of ut: (batch, u_dim)
            c_hist = torch.zeros(0).cuda().double()  # Initialize context vectors history (empty). Must be of size (t - 1, batch, c_dim)
        else:
            ut = torch.zeros(sequence.size()[1], self.u_dim).double()
            c_hist = torch.zeros(0).double()

        for t, xt in enumerate(sequence): # xt is of size (batch, input_dim)
            ut, ct = self.hri_cell(xt, ut, c_hist)

            # Update context history
            c_hist = torch.cat([c_hist, ct.unsqueeze(0)], 0)

        return ut

class ClassificationModule(nn.Module):
    """
    HRI-RNN or baseline GRU followed by a dense layer (MLP) for classification
    (signs of user engagement decrease detection)
    """
    def __init__(self, input_dim, hidden_dims=None, architecture='SimpleRNN', use_gpu=False, user_data_only=False, attend_over_context=0):
        # input_dim: dimension of the feature space
        # hidden_dims: dimensions of the embedding spaces (RNN states)
        super(ClassificationModule, self).__init__()
        self.use_gpu = use_gpu
        self.architecture = architecture

        # Sequence embedding module
        if architecture == 'HriRNN':
            assert len(hidden_dims) == 2
            self.module = HriRNN(input_dim, hidden_dims[0], hidden_dims[1], self.use_gpu, attend_over_context=attend_over_context)

            # Linear mapping that projects the data into a 1D space (binary classification)
            self.dense_layers = MLP(hidden_dims[0], hidden_dims[0] // 2, 1)
        else:
            # By default, the baseline GRU (SimpleRNN) is run
            self.module = SimpleRNN(input_dim, hidden_dims[0], user_data_only=user_data_only, attend_over_context=attend_over_context)
            self.dense_layers = MLP(hidden_dims[0], hidden_dims[0] // 2, 1)

    def forward(self, sequence):
        """
        :param sequence: Sequence of data frames of size (batch, seq_len, input_dim)
        :return: ut: Last user state in the sequence of size (batch, u_dim)
        """
        # TODO:
        #  - Return attention weights

        seq_representation = self.module(sequence)
        res = self.dense_layers(seq_representation)
        res = torch.sigmoid(res)
        return res