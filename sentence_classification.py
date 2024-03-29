"""Dataloader for language generation"""
from collections import Counter
from itertools import chain

import numpy as np

from data_loader import ClassificationBase

# pylint: disable=W0223


class SentenceClassification(ClassificationBase):
	r"""Base class for sentence classification datasets. This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	ARGUMENTS = ClassificationBase.ARGUMENTS
	ATTRIBUTES = ClassificationBase.ATTRIBUTES

	def get_batch(self, key, index):
		'''Get a batch of specified `index`.

		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index

		Returns:
			(dict): A dict at least contains:

				* sent_length(:class:`numpy.array`): A 1-d array, the length of sentence in each batch.
				  Size: `[batch_size]`
				* sent(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Only provide valid words. `unk_id` will be used if a word is not valid.
				  Size: `[batch_size, max(sent_length)]`
				* label(:class:`numpy.array`): A 1-d array, the label of sentence in each batch.
				* sent_allvocabs(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Provide both valid and invalid words.
				  Size: `[batch_size, max(sent_length)]`

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1, 2])
			{
				"sent": numpy.array([
						[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
						[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
						[2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
					]),
				"label": numpy.arrat([1, 2, 0]) # label of sentences
				"sent_length": numpy.array([5, 3, 6]), # length of sentences
				"sent_allvocabs": numpy.array([
						[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
						[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
						[2, 7, 8, 9, 10, 3]   # third sentence: <go> hello i am fine <eos>
					]),
			}
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(index)
		res["sent_length"] = np.array( \
			list(map(lambda i: len(self.data[key]['sent'][i]), index)))
		res_sent = res["sent"] = np.zeros( \
			(batch_size, np.max(res["sent_length"])), dtype=int)
		res["label"] = np.zeros(batch_size, dtype=int)
		for i, j in enumerate(index):
			sentence = self.data[key]['sent'][j]
			res["sent"][i, :len(sentence)] = sentence
			res["label"][i] = self.data[key]['label'][j]

		res["sent_allvocabs"] = res_sent.copy()
		res_sent[res_sent >= self.valid_vocab_len] = self.unk_id
		return res


class SST(SentenceClassification):
	'''A dataloader for preprocessed SST dataset.

	Arguments:
			file_path (str): a str indicates the path of SST dataset.
			valid_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
					not less than `min_vocab_times` in **training set** will be marked as valid words.
					Default: 10.
			max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
					to first `max_sen_length` tokens. Default: 50.
			invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
					not less than `invalid_vocab_times` in the **whole dataset** (except valid words) will be
					marked as invalid words. Otherwise, they are unknown words, both in training or
					testing stages. Default: 0 (No unknown words).

	Refer to :class:`SentenceClassification` for attributes and methods.

	References:
		[1] http://images.cocodataset.org/annotations/annotations_trainval2017.zip

		[2] Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context. ECCV 2014.

	'''

	def __init__(self, file_path='./data', emb_path='./data/vector.txt', \
		min_vocab_times=5, max_sen_length=50, invalid_vocab_times=0):
		self._file_path = file_path
		self._emb_path = emb_path
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._invalid_vocab_times = invalid_vocab_times
		super(SST, self).__init__()

	def _load_data(self):
		r'''Loading dataset, invoked by `LanguageGeneration.__init__`
		'''
		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/%s.txt" % (self._file_path, key))
			origin_data[key] = {}
			_origin_data = list( \
				map(lambda line: (int(line[0]), line[2:].lower().split()), f_file.readlines()))
			origin_data[key]['sent'] = list( \
				map(lambda line: line[1], _origin_data))
			origin_data[key]['label'] = list( \
				map(lambda line: line[0], _origin_data))

		# build vocab
		vocab_list = []
		# load pretrained embedding
		self.emb = []
		with open(self._emb_path, 'r') as r:
			for line in r:
				line = line.split()
				word = line[0].lower()
				_emb = [float(x) for x in line[1:]]
				vocab_list.append(word)
				self.emb.append(_emb)
		self.emb = np.array(self.emb)
		vocab_list = self.ext_vocab + vocab_list
		self.emb = np.concatenate([np.random.random((len(self.ext_vocab), 300)) - .5, self.emb], axis=0)
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		raw_vocab_list = []
		for key in self.key_name:
			raw_vocab_list.extend(list(chain(*(origin_data[key]['sent']))))
		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list( \
			filter( \
				lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, \
				vocab))
		vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

		print("valid vocab list length = %d" % valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))

		word2id = {w: i for i, w in enumerate(vocab_list)}
		def line2id(line):
			return ([self.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) \
					+ [self.eos_id])[:self._max_sen_length]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}
			data[key]['sent'] = list(map(line2id, origin_data[key]['sent']))
			data[key]['label'] = origin_data[key]['label']
			data_size[key] = len(data[key]['sent'])

			vocab = list(chain(*(origin_data[key]['sent'])))
			vocab_num = len(vocab)
			oov_num = len( \
				list( \
					filter( \
						lambda word: word not in word2id, \
						vocab)))
			invalid_num = len( \
				list( \
					filter( \
						lambda word: word not in valid_vocab_set, \
						vocab))) - oov_num
			length = list( \
				map(len, origin_data[key]['sent']))
			cut_num = np.sum( \
				np.maximum( \
					np.array(length) - \
					self._max_sen_length + \
					1, \
					0))
			print( \
				"%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, cut word rate: %f" % \
				(key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, valid_vocab_len, data, data_size


if __name__ == '__main__':
	sst = SST()
	print('embedding shape: ', sst.emb.shape)
