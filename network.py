# coding:utf-8
import logging

import torch
from torch import nn

# pylint: disable=W0221


class Network(nn.Module):
	def __init__(self, emb, rnn_size=200, mode='GRU'):
		super(Network, self).__init__()
		'''
		mode: 'GRU', 'LSTM', 'Attention'
		'''
		self.mode = mode

		self.embLayer = EmbeddingLayer(emb)
		self.encoder = Encoder(embedding_size=emb.shape[1], rnn_size=rnn_size, mode=mode)
		if mode == 'Attention':
			self.selfAttention = SelfAtt(hidden_size=rnn_size)
		self.predictionNetwork = PredictionNetwork(rnn_size=rnn_size)

		self.loss = nn.CrossEntropyLoss()

	def forward(self, sent, sent_length, label=None):

		embedding = self.embLayer.forward(sent)
		hidden_states = self.encoder.forward(embedding, sent_length)
		if self.mode == 'Attention':
			sentence_representation, penalization_loss = self.selfAttention.forward(hidden_states)
		else:
			sentence_representation = hidden_states.mean(dim=1)
		logit = self.predictionNetwork.forward(sentence_representation)

		if label is None:
			return logit

		classification_loss = self.loss(logit, label)
		if self.mode == 'Attention':
			return logit, classification_loss + penalization_loss * .0
		else:
			return logit, classification_loss


class EmbeddingLayer(nn.Module):
	def __init__(self, emb):
		super(EmbeddingLayer, self).__init__()

		vocab_size, embedding_size = emb.shape
		self.embLayer = nn.Embedding(vocab_size, embedding_size)
		self.embLayer.weight = nn.Parameter(torch.Tensor(emb))

	def forward(self, sent):
		'''
		inp: data
		output: post
		'''
		return self.embLayer(sent)


class LSTM(nn.Module):
	"""docstring for LSTM"""
	def __init__(self, input_size, hidden_size):
		super(LSTM, self).__init__()
		# TODO: Implement LSTM
		self.f_wise = nn.Linear(input_size + hidden_size, hidden_size)
		self.i_wise = nn.Linear(input_size + hidden_size, hidden_size)
		self.o_wise = nn.Linear(input_size + hidden_size, hidden_size)
		self.c_wise = nn.Linear(input_size + hidden_size, hidden_size)

		self.hidden_size = hidden_size

	def forward(self, embedding, init_h=None, inin_c=None):
		'''
		embedding: [sentence_length, batch_size, embedding_size]
		init_h   : [batch_size, hidden_size]
		'''
		# TODO: Implement LSTM
		sentence_length, batch_size, embedding_size = embedding.size()
		if init_h is None:
			h = torch.zeros(batch_size, self.hidden_size, dtype=embedding.dtype, device=embedding.device)
		else:
			h = init_h
		if inin_c is None:
			c = torch.zeros(batch_size, self.hidden_size, dtype=embedding.dtype, device=embedding.device)
		else:
			c = inin_c
		hidden_states = []
		for t in range(sentence_length):
			_input = torch.cat([embedding[t], h], dim=1)
			f = torch.sigmoid(self.f_wise(_input))
			i = torch.sigmoid(self.i_wise(_input))
			o = torch.sigmoid(self.o_wise(_input))
			c = f*c + i*torch.tanh(self.c_wise(_input))
			h = o * torch.tanh(c)

			hidden_states.append(h)

		return torch.stack(hidden_states, dim=1)


class GRU(nn.Module):
	"""docstring for GRU"""
	def __init__(self, input_size, hidden_size):
		super(GRU, self).__init__()
		self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.memory_gate = nn.Linear(input_size + hidden_size, hidden_size)

		self.hidden_size = hidden_size

	def forward(self, embedding, init_h=None):
		'''
		embedding: [sentence_length, batch_size, embedding_size]
		init_h   : [batch_size, hidden_size]
		'''
		sentence_length, batch_size, embedding_size = embedding.size()
		if init_h is None:
			h = torch.zeros(batch_size, self.hidden_size, \
							dtype=embedding.dtype, device=embedding.device)
		else:
			h = init_h
		hidden_states = []
		for t in range(sentence_length):
			_input = torch.cat([embedding[t], h], dim=1)
			z = torch.sigmoid(self.update_gate(_input)) # [batch_size, hidden_size]
			r = torch.sigmoid(self.reset_gate(_input)) # [batch_size, hidden_size]
			# TODO: Update hidden state h
			temp = torch.cat([embedding[t], r*h], dim=1)
			_h = torch.tanh(self.memory_gate(temp)) # [batch_size, hidden_size]
			h = (1-z)*h+z*_h
			hidden_states.append(h)

		return torch.stack(hidden_states, dim=1) # [batch_size, sentence_length, hidden_size]


class Encoder(nn.Module):
	def __init__(self, embedding_size, rnn_size, mode='GRU'):
		super(Encoder, self).__init__()

		if mode == 'GRU':
			self.rnn = GRU(embedding_size, rnn_size)
		else:
			self.rnn = LSTM(embedding_size, rnn_size)

	def forward(self, embedding, sent_length=None):
		'''
		sent_length is not used
		'''
		hidden_states = self.rnn(embedding.transpose(0, 1)) # [batch_size, sentence_length, hidden_size]
		# you can add dropout here
		return hidden_states


class SelfAtt(nn.Module):
	"""docstring for SelfAtt"""
	def __init__(self, hidden_size):
		super(SelfAtt, self).__init__()
		# TODO: Implement Self-Attention

		self.ws1 = torch.nn.Linear(hidden_size, hidden_size, False)
		self.ws2 = torch.nn.Linear(hidden_size, 1, False)
		# self.l_s = torch.nn.functional.softmax(1)

	def forward(self, h, add_penalization=True):
		'''
		h: [batch_size, sentence_length, hidden_size]
		'''
		# TODO: Implement Self-Attention
		A = torch.transpose(torch.nn.functional.softmax(self.ws2(torch.tanh(self.ws1(h))), dim = 1) , 1 , 2)
		M = torch.bmm(A, h)
		M = torch.squeeze(M)
		batch_size = h.shape[0]
		I = torch.ones(batch_size, 1, 1).cuda()
		P = torch.norm(torch.bmm(A, torch.transpose(A, 1, 2)) - I) ** 2
		return M, P


class PredictionNetwork(nn.Module):
	def __init__(self, rnn_size, hidden_size=64, class_num=5):
		super(PredictionNetwork, self).__init__()
		self.predictionLayer = nn.Sequential(nn.Linear(rnn_size, hidden_size),
											nn.ReLU(),
											nn.Linear(hidden_size, class_num))

	def forward(self, h):

		return self.predictionLayer(h)
