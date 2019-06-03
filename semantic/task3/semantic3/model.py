# -*- coding: utf-8 -*- 
# Author: Shengqiang Zhang
# Time  : 2019/5/14 21:33
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        return {"token_to_idx": self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.

                Args:
                    token (str): the item to add into the Vocabulary
                Returns:
                    index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError(f"the index {index} not in the vocabulary")
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % (len(self))

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>", mask_token="<MASK>",
                 begin_seq_token="<BEGIN>", end_seq_token="<END>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)
        self._unk_token = unk_token
        self._mask_token = mask_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        # self.unk_index = self.add_token(self._unk_token)
        self.unk_index = -1
        self.mask_index = self.add_token(self._mask_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({"unk_token": self._unk_token,
                         "mask_token": self._mask_token,
                         "begin_seq_token": self._begin_seq_token,
                         "end_seq_token": self._end_seq_token})
        return contents

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class Vectorizer:
    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        if vector_length <= 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector

    def _get_source_indices(self, text):
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(self.source_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.source_vocab.end_seq_index)

        return indices

    def _get_target_indices(self, text):
        indices = [self.target_vocab.lookup_token(token) for token in text.split(" ")]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_dataset_max_length=True):
        source_vector_length = -1
        target_vector_length = -1
        if use_dataset_max_length:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices, source_vector_length, self.source_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices, target_vector_length, self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices, target_vector_length, self.target_vocab.mask_index)

        return {"source_vector": source_vector,
                "target_x_vector": target_x_vector,
                "target_y_vector": target_y_vector,
                "source_length": len(source_indices)}

    @classmethod
    def from_dataframe(cls, bitext_df):
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()

        max_source_length = 0
        max_target_length = 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])

        return cls(source_vocab, target_vocab,
                   contents["max_source_length"], contents["max_target_length"])

    def to_serializable(self):
        return {"source_vocab": self.source_vocab.to_serializable(),
                "target_vocab": self.target_vocab.to_serializable(),
                "max_source_length": self.max_source_length,
                "max_target_length": self.max_target_length}


class Dataset(Dataset):
    def __init__(self, text_df, vectorizer):
        self.text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df.split == "train"]
        self.train_size = len(self.train_df)

        self.dev_df = self.text_df[self.text_df.split == "dev"]
        self.dev_size = len(self.dev_df)

        self.test_df = self.text_df[self.text_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "dev": (self.dev_df, self.dev_size),
            "test": (self.test_df, self.test_size),
        }

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        text_df = pd.read_csv(dataset_csv)
        # train_subset = text_df[text_df.split == "train"]

        return cls(text_df, Vectorizer.from_dataframe(text_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_csv, vectorizer_filepath):
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return Vectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """
        the primary entry point method for PyTorch dataset
        :param index: the index to the data point
        :return: a dictionary hold the data point : (x_data, y_target, class_index)
        """
        row = self._target_df.iloc[index]
        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language)

        return {"x_source": vector_dict["source_vector"],
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"],
                "x_source_length": vector_dict["source_length"]}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_nmt_batches(dataset, batch_size, shuffle=True,
                         drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)
    for data_dict in dataloader:
        lengths = data_dict["x_source_length"].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        super(Encoder, self).__init__()
        self.source_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_size,
                                             padding_idx=0)
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size,
                            bidirectional=True, batch_first=True)

    def forward(self, x_source, x_lengths):
        """

        :param x_source: the input data tensor. shape = (batch_size, seq_size)
        :param x_lengths: a vector for length of each item in the batch
        :return:
            x_unpacked: shape = (batch_size, seq_size, rnn_hidden_size * 2)
            x_birnn_h: shape = (batch_size, rnn_hidden_size * 2)
        """
        x_embedded = self.source_embedding(x_source)

        x_packed = pack_padded_sequence(input=x_embedded,
                                        lengths=x_lengths.detach().cpu().numpy(),
                                        batch_first=True)

        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
        # from pytorch doc: num_rnn = num_layers * num_directions
        #                   feature_size = num of features in hidden state
        x_birnn_out, x_birnn_h = self.birnn(x_packed)

        # permute to (batch_size, num_rnn, feature_size)
        x_birnn_h = x_birnn_h.permute(1, 0, 2)
        # reshape to (batch_size, num_rnn * feature_size)
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)

        x_unpacked, _ = pad_packed_sequence(sequence=x_birnn_out,
                                            batch_first=True)
        return x_unpacked, x_birnn_h


def verbose_attention(encoder_state_vectors, query_vector):
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    attention_score = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size),
                                dim=2)
    attention_distribution = F.softmax(attention_score, dim=1)
    weighted_vectors = encoder_state_vectors * attention_distribution.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)

    return context_vectors, attention_distribution, attention_score


def terse_attention(encoder_state_vectors, query_vector):
    attention_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    attention_distribution = F.softmax(attention_scores, dim=-1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1),
                                   attention_distribution.unsqueeze(dim=2)).squeeze()
    return context_vectors, attention_distribution


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index):
        super(Decoder, self).__init__()
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_size,
                                             padding_idx=0)
        self._rnn_hidden_size = rnn_hidden_size
        self.gru_cell = nn.GRUCell(input_size=embedding_size + rnn_hidden_size,
                                   hidden_size=rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)
        self.bos_index = bos_index
        self._sampling_temperature = 3

    def _init_indices(self, batch_size):
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index

    def _init_context_vectors(self, batch_size):
        return torch.zeros(batch_size, self._rnn_hidden_size)

    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0.0):
        """

        :param sample_probability: the schedule sampling parameter probability of
                                    using model's predictions at each decoder step
        :param encoder_state: the output of the encoder output
        :param initial_hidden_state: the last hidden state in the Encoder
        :param target_sequence: the target text data tensor
        :return:
            output_vectors: prediction vectors at each output step
        """
        # target_sequence is shape(batch_size, seq_size), we want shape(seq_size, batch_size)
        if target_sequence is None:
            sample_probability = 1.0
        else:
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)

        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)

        context_vectors = self._init_context_vectors(batch_size)
        # Initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)

        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []

        self._cache_p_attn = []
        self._cache_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        for i in range(output_sequence_size):
            use_sample = np.random.random() < sample_probability  # todo
            if not use_sample:
                y_t_index = target_sequence[i]

            # Step 1: Embed word and concat with previous context
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat((y_input_vector, context_vectors), dim=1)

            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            self._cache_ht.append(h_t.cpu().detach().numpy())

            # Step 3: Use the current hidden to attend to the encoder state
            context_vectors, attention_distribution, attention_score = \
                verbose_attention(encoder_state_vectors=encoder_state,
                                  query_vector=h_t)

            # cache the attention distribution for visualization
            self._cache_p_attn.append(attention_distribution.cpu().detach().numpy())

            # Step 4: use the current hidden state and context vectors to predict next word
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(input=prediction_vector, p=0.3))

            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()

            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        return output_vectors


class Model(nn.Module):
    """ The Neural Machine Translation Model """

    def __init__(self, source_vocab_size, source_embedding_size,
                 target_vocab_size, target_embedding_size, encoding_size,
                 target_bos_index):
        """
        Args:
            source_vocab_size (int): number of unique words in source language
            source_embedding_size (int): size of the source embedding vectors
            target_vocab_size (int): number of unique words in target language
            target_embedding_size (int): size of the target embedding vectors
            encoding_size (int): the size of the encoder RNN.
            target_bos_index (int): index for BEGIN-OF-INDEX token
        """
        super(Model, self).__init__()
        self.encoder = Encoder(num_embeddings=source_vocab_size,
                               embedding_size=source_embedding_size,
                               rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder = Decoder(num_embeddings=target_vocab_size,
                               embedding_size=target_embedding_size,
                               rnn_hidden_size=decoding_size,
                               bos_index=target_bos_index)

    def forward(self, x_source, x_source_lengths,
                target_sequence, sample_probability=0.0):
        """The forward pass of the model

        Args:
            x_source (torch.Tensor): the source text data tensor.
                x_source.shape should be (batch, vectorizer.max_source_length)
            x_source_lengths torch.Tensor): the length of the sequences in x_source
            target_sequence (torch.Tensor): the target text data tensor
        Returns:
            decoded_states (torch.Tensor): prediction vectors at each output step
            sample_probability:
        """
        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoded_states = self.decoder(encoder_state=encoder_state,
                                      initial_hidden_state=final_hidden_states,
                                      target_sequence=target_sequence,
                                      sample_probability=sample_probability)
        return decoded_states
