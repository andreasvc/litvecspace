"""Divide corpus into chunks and get BoW and DBoW representations."""
import os
import glob
import logging
import random
from collections import Counter
import pandas
import gensim
import numpy as np
import scipy.sparse
from sklearn import feature_extraction
import joblib

# General parameters
CORPUSCHUNKS = '/datastore/avcrane1/Riddle/riddleallchunks1000word.txt'
CORPUSLEMMACHUNKS = '/datastore/avcrane1/Riddle/riddleallchunks1000lemma.txt'
CHUNKNAMESFILE = '/datastore/avcrane1/Riddle/riddleallchunknames1000word.txt'
CHUNKLEN = 1000  # tokens (including punctuation)
NUMCORES = 16  # number of CPU cores to use
SEED = 42  # fix random seed

# doc2vec parameters
D2VDIM = 300
D2VWINDOW = 10
D2VMINCOUNT = 10
D2VSAMPLETHRESH = 1e-5
D2VNEGATIVE = 5
D2VPASSES = 10
D2VMODE = 0  # 0 = dbow; 1 = dmpv
D2VPATH = '/datastore/avcrane1/deeplit.new/doc2vecmodel.bin'

# enable logging
logging.basicConfig(
		format='%(asctime)s : %(levelname)s : %(message)s',
		level=logging.INFO)


def splitcorpus():
	"""Chunkize corpus."""
	chunknames = []
	with open(CORPUSCHUNKS, 'w', encoding='utf8') as out1:
		with open(CHUNKNAMESFILE, 'w', encoding='utf8') as out2:
			for filename in glob.glob(
					'/datastore/avcrane1/Riddle/tokenized_sentno/*.tok'):
				# drop last chunk, because it may be much shorter
				for name, chunk in list(chunkize(filename, CHUNKLEN))[:-1]:
					out1.write(' '.join(chunk).lower() + '\n')
					out2.write(name + '\n')
					chunknames.append(name)

	# lemma unigrams
	# filter punctuation + names
	FUNCWORDPOS = {
		'let',  # punctuation
		'lid',  # determiner
		'vnw',  # pronoun
		'vg',   # conjuction
		'vz'    # preposition
		'tsw',  # numeral
		'spec', # part of proper noun
		}
	STOPWORDS = set(
		'zijn hebben worden zullen moeten weten zien komen zitten staan '
		'denken kunnen willen zeggen gaan vinden laten vragen maken vinden '
		'de het een deze die dit dat daar er als dan ander over heen weer ge '
		'niet niets geen wel te toch nog en maar goed doch bij der altijd '
		'haar ze mijn zonder naar doen omdat we iemand men alleen '
		'met ja toen om tegen of kon voor iets hier veel op wie zelf '
		'wil wij zo ons van eens tot hem wat door hun waarom '
		'ook me dus ben zij uw aan hij je meer alles reeds af al ik '
		'uit want in hoe na nu nou mij zich u'.split())

	curfile = None
	inp = None
	with open(CORPUSLEMMACHUNKS, 'w', encoding='utf8') as out:
		for chunk in chunknames:
			fname, sentno = chunk.split(' ', 1)
			doc = []
			if curfile != fname:
				if inp is not None:
					inp.close()
				inp = open(
						'/datastore/avcrane1/Riddle/parses/%s.export' % fname,
						encoding='utf8')
				curfile = fname
			end = '#EOS %s/%s' % (fname, sentno)
			line = None
			while line != end:
				line = inp.readline().rstrip()
				if not line.startswith('#'):
					fields = line.split()
					if (fields[2] not in FUNCWORDPOS
							and not fields[3].startswith('N(eigen,')
							and fields[1] not in STOPWORDS):
						doc.append(fields[1])
			out.write(' '.join(doc) + '\n')
	inp.close()


def chunkize(filename, n):
	"""Make chunks with `n` words rounded to nearest sentence boundary."""
	label = os.path.splitext(os.path.basename(filename))[0]
	tmp = []
	words = 0
	prevsentno = ''
	with open(filename, encoding='utf8') as inp:
		for line in inp:
			sentno, line = line.rstrip().split('|', 1)
			lensent = line.count(' ') + 1
			if words + lensent < n:
				tmp.append(line)
				words += lensent
			elif n - words < words + lensent - n:
				name = '%s %s' % (label, prevsentno)
				yield name, tmp
				tmp = [line]
				words = lensent
			else:
				name = '%s %s' % (label, sentno)
				yield name, tmp + [line]
				tmp = []
				words = 0
			prevsentno = sentno
		if tmp:
			name = '%s %s' % (label, sentno)
			yield name, tmp


def readchunks():
	docs = []
	docslemma = []
	y = []
	df = pandas.read_csv('/datastore/avcrane1/Riddle/metadata.csv', index_col=0)
	with open(CHUNKNAMESFILE, encoding='utf8') as inp:
		chunknames = inp.read().splitlines()
	with open(CORPUSCHUNKS, encoding='utf8') as inp:
		with open(CORPUSLEMMACHUNKS, encoding='utf8') as inp1:
			for chunkname, line, line1 in zip(chunknames, inp, inp1):
				doc = gensim.models.doc2vec.TaggedDocument(
						line.strip().split(),
						[chunkname])
						# [chunkname.split(' ')[0], chunkname])
				docs.append(doc)
				docslemma.append(line1.strip().split())
				y.append(df.at[chunkname.split(' ')[0], 'Literary rating'])
	index = {doc.tags[0]: doc for doc in docs}
	y = np.array(y)
	return docs, docslemma, y, index


def getbow(docs, docslemma):
	"""Unigram/bigram BoW baseline."""
	# Restrict vocabulary to avoid memory problems with CountVectorizer
	# Get 100,000 MFW
	vocab = [word for word, _ in Counter(word
			for doc in docs
				for word in doc.words
				).most_common(100000)]
	vectorizer = feature_extraction.text.CountVectorizer(
		tokenizer=dummy, preprocessor=dummy, binary=False,
		ngram_range=(1, 1), vocabulary=vocab)
	x_uni = vectorizer.fit_transform(doc.words for doc in docs)
	store_sparse_mat('unigram.npz', scipy.sparse.csc_matrix(x_uni),
			[doc.tags[0] for doc in docs], vectorizer.get_feature_names())
	joblib.dump(vectorizer, 'unigramvectorizer.pkl')

	# Get 100,000 most frequent bigrams
	bigramvocab = [bigram for bigram, _ in Counter(word1 + ' ' + word2
			for doc in docs
				for word1, word2 in zip(doc.words, doc.words[1:])
				).most_common(100000)]
	vectorizer = feature_extraction.text.CountVectorizer(
		tokenizer=dummy, preprocessor=dummy, binary=False,
		ngram_range=(2, 2), vocabulary=bigramvocab)
	x_bi = vectorizer.fit_transform(doc.words for doc in docs)
	store_sparse_mat('bigram.npz', scipy.sparse.csc_matrix(x_bi),
			[doc.tags[0] for doc in docs], vectorizer.get_feature_names())
	joblib.dump(vectorizer, 'bigramvectorizer.pkl')

	return x_uni


def traindoc2vec(docs):
	"""Create DBoW paragraph vector model."""
	alpha, min_alpha = 0.025, 0.0001
	random.seed(SEED)
	shuffleddocs = docs[:]
	random.shuffle(shuffleddocs)

	# train the doc2vec model
	doc2vecmodel = gensim.models.Doc2Vec(
			shuffleddocs,
			vector_size=D2VDIM, window=D2VWINDOW, min_count=D2VMINCOUNT,
			sample=D2VSAMPLETHRESH, hs=0, dm=D2VMODE, negative=D2VNEGATIVE,
			epochs=D2VPASSES, seed=SEED, workers=NUMCORES,
			dm_concat=0, dbow_words=1, alpha=alpha, min_alpha=min_alpha)
	doc2vecmodel.save(D2VPATH)
	pandas.DataFrame(
			[doc2vecmodel.infer_vector(doc.words) for doc in docs],
			index=[doc.tags[0] for doc in docs]).to_csv('d2v.csv')


def store_sparse_mat(filename, data, rows, columns):
	"""Store scipy.sparse csc matrix in .npz format,
	along with corresponding row/column labels."""
	if data.__class__ != scipy.sparse.csc.csc_matrix:
		raise ValueError('This code only works for csc matrices')
	np.savez_compressed(filename,
			data=data.data,
			indices=data.indices,
			indptr=data.indptr,
			shape=data.shape,
			rows='\n'.join(rows).encode('utf8'),
			columns='\n'.join(columns).encode('utf8'))


def load_sparse_mat(filename):
	"""Load scipy.sparse csc matrix stored as numpy arrays."""
	inp = np.load(filename)
	data = scipy.sparse.csc_matrix(
			(inp['data'], inp['indices'], inp['indptr']),
			shape=inp['shape'], dtype=np.int32)
	rows = pandas.Index(bytes(inp['rows']).decode('utf8').splitlines())
	columns = pandas.Index(bytes(inp['columns']).decode('utf8').splitlines())
	return data, rows, columns


def dummy(x):
	"""A function that does nothing; separate function needed for pickle."""
	return x


def main():
	splitcorpus()
	docs, docslemma, _y, _index = readchunks()
	getbow(docs, docslemma)
	traindoc2vec(docs)

if __name__ == '__main__':
	main()
