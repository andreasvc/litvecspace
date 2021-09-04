#!/usr/bin/env python
# coding: utf-8
"""Train and save predictive regression model based on paragraph vectors (d2v).
"""
# std lib
import os
from functools import reduce
from operator import or_
# pip
import numpy as np
import pandas
# import gensim
from sklearn import metrics, linear_model
import joblib


def evalreport(y_true, y_pred, folds, names, mask=None):
	"""Given two Series objects and a sequence 'folds', compute metrics for
	each fold and report mean/stderr."""
	if mask is None:
		mask = np.ones(len(names), dtype=bool)
	result = pandas.DataFrame(index=['mean', 'std err'])
	# NB: calculate mean of MSE of each fold, then take root of overall
	# mean. http://stats.stackexchange.com/a/85517
	mse = pandas.Series(
			[metrics.mean_squared_error(y_true[a & mask], y_pred[a & mask])
			for a in folds])
	r2 = pandas.Series([metrics.r2_score(y_true[a & mask], y_pred[a & mask])
			for a in folds])
	tau = pandas.Series(
			[y_true[a & mask].corr(y_pred[a & mask], method='kendall')
			for a in folds])
	result['$R^2$'] = 100 * pandas.Series([
			r2.mean(), r2.sem()], index=result.index)
	result[r'Kendall $\tau$'] = pandas.Series([
			tau.mean(), tau.sem()], index=result.index)
	result['RMS error'] = (pandas.Series([
			mse.mean(), mse.sem()], index=result.index) ** 0.5)
	return result.T


def evalpred(predmodel, data, y, names, folds, mask=None):
	"""Fit models and evaluate with crossvalidation."""
	if mask is None:
		mask = np.ones(len(names), dtype=bool)
	y_pred = pandas.Series(index=names, name='prediction', dtype=np.float64)
	for n, fold in enumerate(folds):
		otherfolds = reduce(or_, [a for m, a in enumerate(folds) if m != n])
		predmodel.fit(data[otherfolds & mask], y[otherfolds & mask])
		y_pred[fold & mask] = predmodel.predict(data[fold & mask])
	return evalreport(y, y_pred, folds, names, mask).round(decimals=3), y_pred


def gettarget(mdfname, names):
	"""Get list of literary ratings corresponding to chunks in `index`."""
	md = pandas.read_csv(mdfname, index_col=0)
	y = []
	for chunkname in names:
		y.append(md.at[chunkname.split(' ')[0], 'Literary rating'])
	y = np.array(y)
	return y


def main():
	# Load data/models
	x_d2v = pandas.read_csv('d2v.csv', index_col=0)
	names = x_d2v.index
	y = gettarget('metadata.csv', names)

	# Metadata: prediction targets
	y = pandas.Series(y, index=names)

	os.chdir('/home/andreas/code/literariness/Riddle')
	target = pandas.read_csv('features/target.csv', index_col=0)
	folds = []  # folds[0]: binary mask of all chunks in fold 0
	# folds[0] can be used as test set, with ~folds[0] equivalent
	# to folds[1:5] as train set.
	# NB: ~folds[0] does include books not assigned to any folds
	# due to rating/#sentences.
	for n in sorted(target.fold.unique()):
		labelsinfold = target[target.fold == n].index
		folds.append(np.array([label.rsplit(' ', 1)[0] in labelsinfold
				for label in names], bool))

	os.chdir('/home/andreas/code/litvecspace')
	# PV DBoW doc2vec
	predmodel = linear_model.RidgeCV(
			alphas=[10.0, 1.0, 1e-1, 1e-3, 1e-4, 1e-6])
	print(evalpred(predmodel, x_d2v, y, names, folds)[0])

	# fit again wo/crossvalidation
	predmodel.fit(x_d2v, y)
	joblib.dump(predmodel, 'd2vpredmodel.pkl')


if __name__ == '__main__':
	main()
