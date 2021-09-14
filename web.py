"""Web interface for applying predictive models to given sample of text."""
# stdlib
from __future__ import print_function, absolute_import
import base64
import io
import os
import sys
import logging
import subprocess
from collections import OrderedDict
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import pandas as pd
import gensim
# Flask & co
from flask import Flask
from flask import request, render_template
import joblib

DEBUG = True  # when True: enable debugging interface, disable multiprocessing
APP = Flask(__name__)
STANDALONE = __name__ == '__main__'
CHUNKLEN = 1000  # tokens (including punctuation)


@APP.route('/')
@APP.route('/index')
def index():
	"""Start page where a text can be entered."""
	return render_template('index.html')


@APP.route('/results', methods=('POST', ))
def results():
	"""Results page for given text."""
	if 'text' not in request.form:
		return 'No form'
	text = request.form['text']
	chunks = list(chunkize(tokenize(text), CHUNKLEN))
	feat = extractfeatures(chunks)
	pred, summary, boxplot, histplot = getpredictions(feat)
	return render_template(
			'predictions.html', chunks=zip(chunks, pred), pred=pred,
			summary=summary, boxplot=boxplot, histplot=histplot)


def tokenize(text):
	# FIXME: more extensive cleaning
	text = text.replace('\n', ' ')
	with subprocess.Popen(
			[os.path.join(
				os.getenv('ALPINO_HOME'), 'Tokenization/tokenize.sh')],
			stdin=subprocess.PIPE, stdout=subprocess.PIPE,
			stderr=subprocess.PIPE) as proc:
		out, err = proc.communicate(text.encode('utf8'))
	return out.decode('utf8')


def chunkize(text, n, label='chunk'):
	"""Make chunks with `n` words rounded to nearest sentence boundary."""
	tmp = []
	words = 0
	chunkno = 0
	for line in text.splitlines():
		lensent = line.count(' ') + 1
		if words + lensent < n:
			tmp.append(line)
			words += lensent
		elif n - words < words + lensent - n:
			chunkno += 1
			name = '%s %s' % (label, chunkno)
			yield name, '\n'.join(tmp)
			tmp = [line]
			words = lensent
		else:
			chunkno += 1
			name = '%s %s' % (label, chunkno)
			yield name, '\n'.join(tmp + [line])
			tmp = []
			words = 0
	if tmp:
		chunkno += 1
		name = '%s %s' % (label, chunkno)
		yield name, '\n'.join(tmp)


def extractfeatures(chunks):
	docs = [gensim.models.doc2vec.TaggedDocument(chunk.split(), [name])
			for name, chunk in chunks]
	result = OrderedDict()
	result['d2v'] = [D2V.infer_vector(doc.words) for doc in docs]
	return result


def b64fig(ax):
	"""Return plot as base64 encoded PNG string for use in data URL."""
	ax.figure.tight_layout()
	figbytes = io.BytesIO()
	ax.figure.savefig(figbytes, format='png')
	return base64.b64encode(figbytes.getvalue()).decode('ascii')


def getpredictions(feat):
	result = MODELS['d2v'].predict(feat['d2v'])
	series = pd.Series(result, name='d2v')
	summary = series.describe().to_string()
	_fig, ax1 = plt.subplots(figsize=(5, 1))
	boxplot = b64fig(series.plot.box(
			xlim=(0.5, 7.99), vert=False, widths=0.7, ax=ax1))
	_fig, ax2 = plt.subplots(figsize=(5, 2))
	histplot = b64fig(series.plot.hist(xlim=(0.5, 7.99), ax=ax2))
	return result, summary, boxplot, histplot


class QueryStringRedirectMiddleware:
	"""Support ; as query delimiter.

	http://flask.pocoo.org/snippets/43/"""
	def __init__(self, application):
		self.application = application

	def __call__(self, environ, start_response):
		qs = environ.get('QUERY_STRING', '')
		environ['QUERY_STRING'] = qs.replace(';', '&')
		return self.application(environ, start_response)


APP.wsgi_app = QueryStringRedirectMiddleware(APP.wsgi_app)

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
log.info('loading.')
if STANDALONE:
	from getopt import gnu_getopt, GetoptError
	try:
		opts, _args = gnu_getopt(sys.argv[1:], '',
				['port=', 'ip=', 'numproc=', 'debug'])
		opts = dict(opts)
	except GetoptError as err:
		print('error: %r' % err, file=sys.stderr)
		sys.exit(2)
	DEBUG = '--debug' in opts
# pre-load data/models
D2V = gensim.models.doc2vec.Doc2Vec.load('doc2vecmodel.bin')
MODELS = dict(d2v=joblib.load('d2vpredmodel.pkl'))
log.info('done.')
if STANDALONE:
	APP.run(use_reloader=True,
			host=opts.get('--ip', '0.0.0.0'),
			port=int(opts.get('--port', 5004)),
			debug=DEBUG)
