import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

class MLP(nn.Module):
    """
    MLP with one hidden layer.
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
    Simple attention mechanism described on the DialogueRNN GitHub page
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
        :param mask: (seq_len, batch). Array of boolean values indicating what token should be considered to compute attention (True)
        """
        Z = self.linear(C).permute(1, 2, 0) # (batch, 1, seq_len)

        # Select sequence elements to which apply attention
        if mask:  # Replace tokens that do not participate in attention computation with -inf
            mask_ = mask.permute(1, 0).unsqueeze(1)  # mask_: (batch, 1, seq_len)
            Z[~mask_] = -float('inf')

        alpha = F.softmax(Z, dim=2) # (batch, 1, seq_len)
        attention_pool = torch.bmm(alpha, C.transpose(0, 1))[:, 0, :] # (batch, dim)

        # TODO: For now, attention weights aren't returned
        return attention_pool # , alpha[:, 0, :] # alpha: (batch, seq_len)

class MatchingAttention(nn.Module):
    """
    General matching attention mechanism described in the DialogueRNN paper
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
        :param mask: (seq_len, batch). Array of boolean values indicating what token should be considered to compute attention (True)
        :return: attention_pool: (batch, mem_dim)
        """
        C_ = C.permute(1, 2, 0) # C_: (batch, mem_dim, seq_len)
        x_ = x.unsqueeze(1) # x_: (batch, 1, input_dim)

        Z = torch.bmm(self.linear(x_), C_) # (batch, 1, seq_len)

        # Select sequence elements to which apply attention
        if mask:  # Replace tokens that do not participate in attention computation with -inf
            mask_ = mask.permute(1, 0).unsqueeze(1)  # mask_: (batch, 1, seq_len)
            Z[~mask_] = -float('inf')

        alpha = F.softmax(Z, dim=2) # (batch, 1, seq_len)
        attention_pool = torch.bmm(alpha, C_.transpose(1, 2))[:, 0, :] # (batch, mem_dim)

        # TODO: For now, attention weights aren't returned
        return attention_pool # , alpha[:, 0, :] # alpha: (batch, seq_len)

class TransformerRNN(nn.Module):
    """
    A recurrent neural network with a Transformer as a first layer.
    Transformer is commented for now.
    """
    def __init__(self, input_dim, hidden_dim):
        super(TransformerRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Transformer layer
        self.encoder_layer = nn.TransformerEncoderLayer(input_dim, 1, dim_feedforward=hidden_dim)
        # Arg. 2: number of self-attention heads
        # Arg. 3: dim. of the feedforward network model of the Transformer encoder (set to default 2048).
        # The feedforward network consists of two sequential linear layers that map the data from
        # input_dim to dim_feedforward, then back to input_dim (quite strange to impose that the embedding
        # and input space have the same dim.)
        
        # Both Transformer and RNN (LSTM, GRU, ...) take as input data of
        # shape (S, N, E) where:
        # - S is the sequence length
        # - N is the batch size
        # - E is the number of features of input data

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 1)
        # Arg. 2: number of encoder layers

        # The RNN takes a sequence with elements in input_dim as inputs and outputs hidden states
        # with dimensionality hidden_dim. Since the Transformer preserves the input data dimension,
        # the RNN receives data of input_dim dimension.
        #
        # RNNs can process data of shape (N, S, E) by setting
        # the parameter batch_first to True
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1)

    def forward(self, sequence):
        # The input sequence is first processed by the Transformer
        # embeds = self.transformer_encoder(sequence)

        embeds = sequence # TODO: Replaced by line above if to use Transformer
        # We use the last hidden state of the last layer of the RNN
        # TODO: In case of LSTM, replace with
        # _, (h, _) = self.rnn(embeds)
        _, h = self.rnn(embeds)
        h = h.permute(1, 0, 2).mean(dim=1) # [:, -1, :]

        # If we want to use the mean of all hidden states, ht, of the LSTM,
        # use the following lines instead
        # h, _ = self.lstm(embeds)
        # h = h.permute(1, 0, 2).mean(dim=1)

        return h.view(-1, self.hidden_dim)

class SimpleRNN(nn.Module):
    """
    A simple GRU with attention.
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
                        data[i][j][32:64] = torch.zeros(64 - 32) # Mask robot audio

        output, h = self.rnn(data)
        # output: (seq_len, batch_size, hidden_dim). All intermediate hidden states
        # h: (1, batch_size, hidden_dim)

        if self.attend_over_context == 1:
            # We apply attention over all intermediate hidden states
            res = self.attention(output)
        elif self.attend_over_context == 0:
            # We return the last hidden state of the GRU's last layer
            h = h.permute(1, 0, 2).mean(dim=1)  # h: (batch_size, 1, hidden_dim)
            res = h.view(-1, self.hidden_dim)  # res: (batch_size, hidden_dim)

        return res

class HriRNNCell(nn.Module):
    def __init__(self, input_dim, u_dim, c_dim, attend_over_context=0):
        super(HriRNNCell, self).__init__()

        self.input_dim = input_dim # Dimension of user data
        self.u_dim = u_dim # Dimension of hidden state of user GRU cell
        self.c_dim = c_dim # Dimension of hidden state of context GRU cell

        self.context_cell = nn.GRUCell(32 + u_dim, c_dim) # 32: number of audio features
        self.user_state_cell = nn.GRUCell(input_dim + c_dim, u_dim) # nn.GRUCell(32 + g_dim, u_dim)

        self.attend_over_context = attend_over_context

        if attend_over_context == 1:
            self.attention = SimpleAttention(self.c_dim)
        elif attend_over_context == 2:
            self.attention = MatchingAttention(self.input_dim, self.c_dim)

    def forward(self, data, u_prev, c_hist, mask=None):
        # data: (batch, input_dim)
        # u_prev: (batch, u_dim)
        # c_hist: (t - 1, batch, c_dim)

        # TODO: Check initializations works fine when on GPU

        # Prepare input for global state and user GRU cells
        data_u =  data.clone()
        data_u[:, 32:64] = torch.zeros(data.size()[0], 64 - 32)

        # TODO: Add default case where the context is updated at each time step
        # Select data where the robot is speaking (over input batch)
        condition = data[:, -1] > 0
        indices_robot_is_speaking = torch.nonzero(condition, as_tuple=True)[0]

        # Create and initialize context vectors tensor. Size: (batch, c_dim)
        c = torch.zeros(data.size()[0], self.c_dim).double() if c_hist.size()[0] == 0 else c_hist[-1]
        c[indices_robot_is_speaking] = self.context_cell(torch.cat((data[indices_robot_is_speaking][:, 32:64], u_prev[indices_robot_is_speaking]), dim=1),
                                                         torch.zeros(indices_robot_is_speaking.size()[0], self.c_dim).double() if c_hist.size()[0] == 0 else
                                                         c_hist[-1][indices_robot_is_speaking])

        if self.attend_over_context:
            # TODO: Here, select tokens for attention
            if c_hist.size()[0] == 0: # TODO: or selected tokens not empty
                # TODO: Try initializing context with c instead of 0
                c_attention = torch.zeros(data.size()[0], self.c_dim).type(data.type()) # (batch, c_dim)
                # alpha = None # For return statement
            else:
                c_attention = self.attention(c_hist, data_u)
            input_c = c_attention
        else:
            input_c = c
        u = self.user_state_cell(torch.cat((data_u, input_c), dim=1), u_prev)

        return u, c # , alpha

class HriRNN(nn.Module):
    def __init__(self, input_dim, u_dim, c_dim, use_gpu=False, attend_over_context=0):
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
        # Initialize hidden states for global state & user GRU cells (t = 0)
        if self.use_gpu:
            ut = torch.zeros(sequence.size()[1], self.u_dim).cuda().double() # size of ut: (batch, u_dim)
            c_hist = torch.zeros(0).cuda().double()  # Initialize context vectors history (empty). Must be of size (t - 1, batch, c_dim)
        else:
            ut = torch.zeros(sequence.size()[1], self.u_dim).double()
            c_hist = torch.zeros(0).double()

        # attention_weights = []

        # Get elements of input sequences where the robot speaks
        mask = sequence[:, :, -1] > 0 # (seq_len, batch)
        for t, xt in enumerate(sequence): # xt is of size (batch, input_dim)
            ut, ct = self.hri_cell(xt, ut, c_hist, mask[:t + 1, :])
            # Update context history
            c_hist = torch.cat([c_hist, ct.unsqueeze(0)], 0)
            # attention_weights.append(alpha)
        return ut # , attention_weights

class ClassificationModule(nn.Module):
    """
    Neural network for (sequence) binary classification.
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
        elif architecture == 'SimpleRNN':
            self.module = SimpleRNN(input_dim, hidden_dims[0], user_data_only=user_data_only, attend_over_context=attend_over_context)
        else: # Simple MLP
            self.module = MLP(input_dim, input_dim // 2, 1)

        # Linear mapping that projects the data into a 1D space (binary classification)
        self.dense_layers = MLP(hidden_dims[0], hidden_dims[0] // 2, 1) # hidden_dims[0] is the size of the user state, ut

    def forward(self, sequence):
        if self.architecture == 'HriRNN':
            # TODO: Return attention weights in future
            seq_representation = self.module(sequence)
            res = self.dense_layers(seq_representation)
        elif self.architecture == 'SimpleRNN':
            seq_representation = self.module(sequence)
            res = self.dense_layers(seq_representation)
            res = torch.sigmoid(res)
            return res
        else:
            # Input data in this case is of shape (batch, seq., input dim.)
            res = self.module(sequence.mean(dim=1)) # Compute the mean of a sequence (along dim. 1) then feed it to MLP for classification
        res = torch.sigmoid(res)
        return res

# Define a dummy dataset class
class ImbalancedDummyDataset(Dataset):
    """
    Generates a dummy data set consisting of 'size' sequences
    of length 'seq_length' whose elements are Gaussian vectors
    of dimension 'dim'.
    """
    def __init__(self, size=1000, seq_length=3, dim=5, alpha=0.1):
        # Factor for generating imbalanced data sets
        # Alpha is the ratio of positive (1) labels
        data_std_1 = torch.randn(int(alpha * size), seq_length, dim)
        data_std_2 = 2 * torch.randn(size - int(alpha * size), seq_length, dim)

        y_std_1 = torch.ones(int(alpha * size))
        y_std_2 = torch.zeros(size - int(alpha * size))

        self.dummy_data = torch.cat((data_std_1, data_std_2))
        self.dummy_labels = torch.cat((y_std_1, y_std_2))

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        return self.dummy_data[index], self.dummy_labels[index]
    
    def __len__(self):
        return len(self.dummy_data)


# Define a dummy dataset class
class AttentionDummyDataset(Dataset):
    """
    Generates a dummy data set consisting of 'size' sequences
    of length 'seq_length' whose elements are uniformly generated
    0/1 vectors of dimension 'dim' and such that the labels depend
    on an element in the sequence.
    """

    def __init__(self, size=1000, seq_length=3, dim=5):
        self.data = torch.rand((size, seq_length, dim)) # torch.randint(0, 2, (size, seq_length, dim), dtype=torch.float)
        self.labels = torch.round(self.data[:, 0, -1]) # the label is the last bit of the second element in the seq. for each seq.

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# Define a HRI dataset class
class HRIDataset(Dataset):
    """
    Creates a HRI dataset using HRI user data.

    Input:
    - data: an array of the shape (nb_seq x seq_length x nb_feat)
    - labels: labels array of corresponding sequences in data
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        assert index < self.__len__(), "Error: index out of range"
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)