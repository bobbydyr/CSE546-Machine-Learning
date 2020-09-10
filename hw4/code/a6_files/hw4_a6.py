import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def collate_fn(batch):
    """
    Create a batch of data given a list of N sequences and labels. Sequences are stacked into a single tensor
    of shape (N, max_sequence_length), where max_sequence_length is the maximum length of any sequence in the
    batch. Sequences shorter than this length should be filled up with 0's. Also returns a tensor of shape (N, 1)
    containing the label of each sequence.

    :param batch: A list of size N, where each element is a tuple containing a sequence tensor and a single item
    tensor containing the true label of the sequence.

    :return: A tuple containing two tensors. The first tensor has shape (N, max_sequence_length) and contains all
    sequences. Sequences shorter than max_sequence_length are padded with 0s at the end. The second tensor
    has shape (N, 1) and contains all labels.
    """
    # print(batch)
    sentences, labels = zip(*batch)
    sequence_lenes = [len(x) for x in sentences]
    max_sequence_length = max(sequence_lenes)
    # print("max_sequence_length: ", max_sequence_length)

    data_matrix = torch.zeros((len(sentences), max_sequence_length))
    for i in range(len(sentences)):
        for j in range(sequence_lenes[i]):
            data_matrix[i, j] = sentences[i][j]
    # print("data_matrix shape: ", data_matrix.shape)

    return (data_matrix, torch.tensor(labels))


class RNNBinaryClassificationModel(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix
        # print("Embed matrix shape: ", embedding_matrix.shape)
        # print("embedding_matrix: ", embedding_matrix[0])

        self.input_size = 50
        self.hidden_dimension = 64
        self.num_layer = 1

        # The num_layer is default to 1 for all of thoes
        self.RNN = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dimension, batch_first=True)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dimension, batch_first=True)
        self.GRU = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dimension, batch_first=True)
        self.linear = nn.Linear(self.hidden_dimension, 1)


    def forward(self, inputs):
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.
        :param inputs: Tensor of shape (N, max_sequence_length) containing N sequences to make predictions for.
        :return: Tensor of predictions for each sequence of shape (N, 1).
        """
        batch_size = inputs.shape[0]

        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_dimension, requires_grad=True)
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_dimension, requires_grad=True)

        x = self.embedding(inputs.long())
        # x, (hn, cn) = self.LSTM(x, (h0, c0))
        # x, hn = self.RNN(x, h0)
        x, hn = self.GRU(x, h0)
        x = x[:,-1,:]
        x = self.linear(x)

        return x

    def loss(self, logits, targets):
        """
        Computes the binary cross-entropy loss.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Binary cross entropy loss between logits and targets as a scalar tensor.
        """
        # print("logits: ", logits)
        # print("targets: ", targets)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.double().reshape((len(targets), 1)))
        return loss


    def accuracy(self, logits, targets):
        """
        Computes the accuracy, i.e number of correct predictions / N.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Accuracy as a scalar tensor.
        """
        logits = logits.double().reshape((1, len(logits)))

        logits[logits >= 0.5] = 1.0
        logits[logits < 0.5] = 0.0
        totl_cor = torch.sum(logits.double() == targets.double())
        # print("tot: ", totl_cor.item())
        accu =  totl_cor.item() / len(logits[0])
        accu = torch.tensor(accu)
        # print("len: ",accu)
        return accu


# Training parameters
TRAINING_BATCH_SIZE = 50
NUM_EPOCHS = 16
LEARNING_RATE = 0.0001

# Batch size for validation, this only affects performance.
VAL_BATCH_SIZE = 128

