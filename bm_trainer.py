"""
Input: file NAME containing pairs of
       geneName pcc_value
       
       integer for the bin_divisor,
       which is the # that the
       total data points will be
       divided by to get the number
       of bins to use in CDF/PDF 
       calculations. Results
       are sensitive to this bin_divisor.

Output: This creates a PNG image 
        NAME.png that shows the bimodal fit

	This will create two gene sets,
	one for each mode (date/party),
	called NAME.date and NAME.party.

	This will also create 2 files
	with the geneName pcc_value
	pairs for each mode (date/party).

	This prints out the mean and stdev
	for each mode.

Example usage:
        python bm_trainer.py data_file 10

	This will create:
	data_file.png, 
	data_file.date, data_file.party,
	data_file.date.vals, data_file.party.vals
"""
from __future__ import with_statement
import logging, optparse, sys
from itertools import izip
from types import FloatType
import nose
import os, random
from scipy.integrate import quad, Inf
from scipy.stats import norm, uniform, kstest, chi2
from scipy.stats.morestats import pdf_moments, boxcox
from scipy.cluster.vq import kmeans
from scipy.optimize import fmin_tnc, brentq

import matplotlib.pyplot as plt

import numpy as N

bin_divisor = 0

def get_bin_divisor(data_length):
	""" This function used to choose a bin divisor,
            but now it just returns the variable
            set in the main function
	"""

	return bin_divisor
#	print 'length', data_length
#	if data_length >= 5000:
#		bin_divisor = 30
#	elif data_length >= 2000:
#		bin_divisor = 20
#	elif data_length >= 1500:
#		bin_divisor = 20
#	elif data_length > 900:
#		bin_divisor = 25
#	else:
#		bin_divisor = 9
#		
#	return bin_divisor

def testInit():
	"""Test that initialzation excepts proper arguements"""
	dist_list = [('norm', (4,6), 0.5),
					('norm', (8,6), 0.5)]
	test_m = m_modal(dist_list)
	
	assert len(test_m.dists) == 2
	assert len(test_m.weights) == 2
	
def testUnpack():
	"""Test that unpacking works correctly"""
	
	t_val = N.array([5,3,0.5,9,6,0.5])
	d_ch = ['norm', 'norm']
	
	d_list, w = unpack(t_val, d_ch)
	
	assert len(d_list) == 2, 'Returned incorrect len d_list: %d' % len(d_list) 
	assert len(w) == 2, 'Returned incorrect len w: %d' % len(d_list)
	
	n1, n2 = d_list
	w1, w2 = w
	
	assert w1 == 0.5, 'Returned incorrect w1: %d' % w1
	assert w2 == 0.5, 'Returned incorrect w1: %d' % w2
	
	assert N.array_equal(n1.stats(),N.array([5,9])), \
			'Returned incorrect d1 stats: %s' % str(n1.stats())
	assert N.array_equal(n2.stats(), N.array([9,36])), \
			'Returned incorrect d2 stats: %s' % str(n2.stats())

def testBiTraining():
	"""Creates a small bi-modal dataset and calls the trainer"""
	
	#create bimodal dist
	n1 = norm(loc = 4, scale = 0.6).rvs(800)
	n2 = norm(loc = 10, scale = 1.0).rvs(200)
	tot = N.concatenate((n1,n2))
	
	cor_mu = [4.0,10.0] 
	cor_w = [0.8,0.2]
	
	tdist, pval = m_modal.train(tot, ['norm', 'norm'])
	
	for dist, cmu, w, cw in izip(tdist.dists, cor_mu,
				     tdist.weights, cor_w):
		mu = dist.stats()[0]
		nose.tools.assert_almost_equal(mu, cmu, places = 0,
				msg = 'mu should be %f but was %f' % (cmu, mu))
		
		nose.tools.assert_almost_equal(w, cw, places = 0, 
				msg = 'w should be %f but was %f' % (cw, w))
				
def testPval():
	"""Test whether the p-val function is reasonable"""
	
	#create bimodal dist
	n1 = norm(loc = 4, scale = 0.6).rvs(800)
	n2 = norm(loc = 10, scale = 1.0).rvs(800)
	tot1 = N.concatenate((n1,n2))
	
	tdist_good1, pval = m_modal.train(tot1, ['norm', 'norm'])
	assert tdist_good1.pval(tot1)[0], \
		'Returned False bi-m trained on bi-m: %s' \
			% str(tdist_good1.pval(tot1))
	
	
	tdist_good2, pval  = m_modal.train(n1, ['norm'])
	assert tdist_good2.pval(n1)[0], \
		'Returned False un-m trained on uni-m: %s' \
			% str(tdist_good1.pval(n1))
	
	
	tdist_bad1, pval = m_modal.train(tot1, ['norm'])
	assert tdist_bad1.pval(tot1)[0] == False, \
		'Returned True for uni-m trained on bi-m: %s' \
			% str(tdist_bad1.pval(tot1))

def testloglikelihood():
	"""Test the loglikelihood function on a small dataset"""
	#a dataset from matlab
	test_data = N.array([-1.18777701646980400,-2.20232071732343830,
	0.98633739100202267,
	-0.51863506634474621,0.32736756408083439,0.23405701284718494,
	0.02146613887909446,-1.00394446674772490,-0.94714606473854135,
	-0.37442919502916561,-1.18588621380852820,-1.05590292352369100,
	1.47247993441991510,0.05574383183784317,-1.21731745370455100,
	-0.04122713368643211,-1.12834386432022860,-1.34927754310249460,
	-0.26110162306162105,0.95346544550481849])
	test_ans = N.array([26.41374323117374000])
	
	test_m = m_modal([('norm', (test_data.mean(), test_data.std()), 1)])
	N.testing.assert_array_almost_equal(test_m.log_likelihood(test_data), test_ans,
				decimal = 1, err_msg = 'Did not produce correct log-likelihood')
	
	
def testAssignVals():
	"""Test the ability to assign values correctly"""
	
	#create bimodal dist
	n1 = norm(loc = 4, scale = 0.6).rvs(800)
	n2 = norm(loc = 500, scale = 1.0).rvs(800)
	tot = N.concatenate((n1,n2))
	
	tdist, pval = m_modal.train(tot, ['norm', 'norm'])
	dist_vals = tdist.assign(tot)
	
	assert N.sum(dist_vals == 0) == 800, \
			'Did not find 800 mode-0 items'
	
	assert N.sum(dist_vals == 1) == 800, \
			'Did not find 800 mode-1 items'

def testMakeFigure():
	"""Test the ability to make a figure"""
	
	#create bimodal dist
	n1 = norm(loc = 4, scale = 2.6).rvs(800)
	n2 = norm(loc = 9, scale = 1.0).rvs(800)
	tot = N.concatenate((n1,n2))
	
	tdist, pval = m_modal.train(tot, ['norm', 'norm'])
	
	#tdist.make_figure()
	
	tdist.make_figure(data = tot, show = False)

def testGatherData():
	"""Test GatherData with generated data"""

	test_vals = norm(loc = 4, scale = 0.6).rvs(800)
	
	simple_col = map(str, test_vals)
	
	out_data = GatherData(iter(simple_col))
	#need to use "almost_equal" since conterting to and from strings causes 
	#some rounding related data-loss
	N.testing.assert_array_almost_equal(out_data, test_vals, decimal = 1,
		err_msg = 'Did not parse simple file data')
	
	multi_col = '\tsomejujnk\n'.join(simple_col)
	out_data = GatherData(iter(multi_col.split('\n')))
	N.testing.assert_array_almost_equal(out_data, test_vals, decimal = 1,
		err_msg = 'Did not parse when in first-col')
	
	mk_str = lambda x: '453.12\t%s\t279\n' % x
	multi_col2 = map(mk_str, simple_col)
	out_data = GatherData(iter(multi_col2), COL = 1)
	N.testing.assert_array_almost_equal(out_data, test_vals, decimal = 1,
		err_msg = 'Did not parse when given COL kwarg')
	
	out_data = GatherData(iter(simple_col + ['lkdfhsd']))
	N.testing.assert_array_almost_equal(out_data, test_vals, decimal = 1,
		err_msg = 'Did not parse when given junk-data')
	
	nose.tools.assert_raises(ValueError, GatherData, 
						iter(simple_col + ['lkdfhsd']), SKIP_JUNK = False)
	
class skewnorm():
	"""
	A simple class to implement the skew-normal distribution
	"""
	
	def __init__(self, loc, scale, skew):
		try:
			self._sknorm = pdf_moments((loc, scale**2, skew))
			self.st = (loc, scale**2, skew)
		except ValueError:
			self._sknorm = pdf_moments((loc, 1e-8, skew))
			self.st = (loc, 1e-8, skew)
		self.pdf = self._sknorm
	def cdf(self, array_obj):
		
		pdf_fun = self._sknorm
		def _cdf(spot):
			val, j = quad(pdf_fun, -Inf, spot)
		
		v_cdf = N.vectorize(_cdf)
		
		return v_cdf(array_obj)
		
	def stats(self):
		return N.array(self.st)
		
		
	
	
class m_modal():
	def __init__(self, *args):
		"""
		Initializes the class.  Should be given a list of tuples:
		[('norm', (norm1_mean, norm1_std), 0.4), 
			('norm', (norm2_mean, norm2_std), 0.6)]
		Can be given:
			'norm', (mean, std), weight
			'skewnorm', (mean, std, skewness), weight
			'uniform', (min, max), weight
			
		"""
		self.dists = []
		self.weights = []
		self.cutoff = None
		self.use_bc = False
		if len(args) == 0:
			return
		
		for name, params, weight in args[0]:
			if name == 'norm':
				self.dists.append(norm(*params))
				self.weights.append(weight)
			elif name == 'skewnorm':
				#pdf_moments takes the VARIANCE ... so must square the std
				params = (params[0], params[1]**2, params[2])
				self.dists.append(pdf_moments(params))
				self.weights.append(weight)
			elif name == 'uniform':
				self.dists.append(uniform(*params))
				self.weights.append(weight)
			else:
				raise KeyError, 'Unknown distribution %s' % name
		assert N.sum(self.weights) == 1, 'sum(weight) != 1'
		
	def treatdata(self, data):
		"""
		Returns the box-cox transformed data if the flag has been set.
		otherwise it will return the data unchanged.
		"""
		#print 'length', len(data)
		if self.use_bc:
			t_data, llmb = boxcox(data)
			
			return t_data
		else:
			return data
		
	def GetParams(self):
		"""
		Returns a tuple of the parameters for each distribution.
		"""
		
		
		return map(lambda x: x.stats(), self.dists)
		
	
	def pdf(self, spots):
		"""
		Returns the pdf evaluated at the desired points.
		"""
		#print self.weights
		vals = N.zeros_like(spots)
		for d, w in izip(self.dists, self.weights):
			vals += d.pdf(spots)*w
			#vals += d.pdf(spots)
			#print 'val', d.pdf(spots)
			
		return vals
			
	def cdf(self, spots):
		"""
		Returns the cdf evaluated at the desired points.
		"""
		vals = N.zeros_like(spots)
		for d, w in izip(self.dists, self.weights):
			vals += d.cdf(spots)*w
			#vals += d.cdf(spots)
			
		return vals
	
	def get_ratio(self, data):
		t_data = self.treatdata(data)
		n_mean = N.mean(t_data)
		n_std = N.std(t_data)
		
		simple_g = m_modal([('norm', (data.mean(), data.std()), 1)])
		simple_g.use_bc = self.use_bc
		#give original data so not transformed twice
		ll_s = -simple_g.log_likelihood(data)
		ll_this = -self.log_likelihood(data)
		
		ratio = 2*(ll_s - ll_this)
		return ratio
	
	def pval(self, data):
		"""
		Determines whether this model id better then a simple gaussian model.
		Uses a KStest on the data and determines the p-val of this distribution
		and a gaussian distribution.
		returns:
			(TF, this_p)
		TF indicates whether this is a reasonable approximation of the data.
		"""
		t_data = self.treatdata(data)
		n_mean = N.mean(t_data)
		n_std = N.std(t_data)
		
		simple_g = m_modal([('norm', (data.mean(), data.std()), 1)])
		simple_g.use_bc = self.use_bc
		#give original data so not transformed twice
		ll_s = simple_g.log_likelihood(data)
		ll_this = self.log_likelihood(data)
		
		ratio = 2*(ll_this - ll_s)
		#print 'chi2', ratio
		
		# 6 degrees of freedom from Adam,
		# based on bimodality in blood paper
		# the real test uses 3
		# from likelihood ratio test in wikipedia
		this_p = 1-chi2.cdf(ratio, 3)
		#print (ll_s, ll_this), ratio, this_p
		TF = this_p > 0.05
		return TF, this_p
	
	def findcutoff(self):
		"""
		Finds the cutoff between 2 normal distributions.
		"""
		d0 = lambda x: self.dists[0].pdf(x)*self.weights[0]
		d1 = lambda x:self.dists[1].pdf(x)*self.weights[1]
		root_fnc = lambda x: d0(x)-d1(x)
		mu0 = self.dists[0].stats()[0]
		mu1 = self.dists[1].stats()[0]
		try:
			self.cutoff = brentq(root_fnc, mu0, mu1)
		except ValueError:
			self.cutoff = (mu1-mu0)/float(2) + mu0
		return self.cutoff
	
	def get_min_pdf(self, spot):
		min_pdf = 10000
		for d, w in izip(self.dists, self.weights):
			min_pdf = min(min_pdf, d.pdf(spot)*w)
		return min_pdf
	
	def get_min_cdf(self, spot):
		min_cdf = 10000
		for d, w in izip(self.dists, self.weights):
			min_cdf = min(min_cdf, d.cdf(spot)*w)
		return min_cdf
	
	def overlap(self):
		"""
		Calculates the overlap of the distributions.  By default when there is
		only one distribution the function return 0.
		
		"""
		
		if len(self.dists) == 1:
			return 0
		pdf_fun = N.vectorize(self.get_min_pdf)
		bins = N.linspace(0,5,100)
		b_w = bins[1] - bins[0]
		pdf_vals = pdf_fun(bins)*b_w
		over = N.sum(pdf_vals)
		return over
		
	
	def log_likelihood(self, data):
		"""
		Calculate the Log-Likelihood of the fitted distribution given
		the data provided.  Larger values indicate MORE likely.
		"""
		#calculate scalar gaussian density
		t_data = self.treatdata(data)
		#pd = self.pdf(t_data)
		#print 'here2', max(pd), min(pd)
		#print 'here',N.log(self.pdf(t_data))
		#print 'here',N.sum(N.log(self.pdf(t_data)), axis = 0)
		return N.sum(N.log(self.pdf(t_data)), axis = 0)
#		return N.log(N.sum(self.pdf(t_data), axis=0))
	
	
	def assign(self, spots):
		"""
		If given an input of locations it will return an index vector.
		"""
		
		t_data = self.treatdata(spots)
		
		#if this is a bimodal then use the "cutoff" method
		if len(self.dists) == 2:
			if self.cutoff == None:
				cutoff = self.findcutoff()
			else:
				cutoff = self.cutoff
			vals = N.zeros_like(t_data)
			vals[t_data > cutoff] = 1
			return vals.astype('int')
		#if its not then use a different method
		vals = N.zeros((len(t_data), len(self.dists)))
		for d, w, ind in izip(self.dists, self.weights, 
								range(len(self.dists))):
			vals[:,ind] = d.pdf(t_data) * w
		
		return vals.argmax(axis = 1)
		
	def make_figure(self, data = None, fig_file = None, ranges = None, 
			show = False, colors = 'bgrmc', plot_overlap = False):
		"""
		Will make a figure of the distribtion.
		"""
		
		
		if ranges == None:
			# this is used
			mu_l, std_l = self.dists[0].stats()
			mu_u, std_u = self.dists[-1].stats()
			ranges = (mu_l - 5*std_l, mu_u + 5*std_u)
		
		if data != None:
			# I need to reset the bins here to reflect
			# the calculations
			t_data = self.treatdata(data)
			bin_divisor = get_bin_divisor(len(t_data))

			n, bins = N.histogram(t_data, new=True,
					      bins = len(t_data)/bin_divisor)
			ranges = (min(ranges[0], bins[0]), max(ranges[1], bins[-1]))
			normed_bins = n.astype('float')/n.sum()
			width = bins[1]-bins[0]
			dist_ind = self.assign(bins)
			color_vals = map(lambda x: colors[x], dist_ind[1:])
			plt.bar(bins[1:], normed_bins, width, color = color_vals)
		else:
			#print 'here'
			#print 'never called'
			n = N.array([])
			bins = N.linspace(ranges[0], ranges[1], 100)
			width = bins[1]-bins[0]
		
		for d, w, c in izip(self.dists, self.weights, colors):
			vals = d.cdf(bins)
			plot_vals = N.diff(vals)*w
			plt.plot(bins[1:], plot_vals, 'k')
		
		vals = self.cdf(bins)
		plot_vals = N.diff(vals)
		plt.plot(bins[1:], plot_vals, 'k-x')
		
		if plot_overlap:
			pdf_vec_fun = N.vectorize(self.get_min_pdf)
			over_pdf = pdf_vec_fun(bins)
			over_cdf = N.cumsum(over_pdf)*width
			# for ind, b_val in izip(range(len(over_cdf)), bins):
				# over_cdf[ind], j = quad(pdf_vec_fun, -Inf, b_val)
			over_vals = N.diff(over_cdf)
			plt.bar(bins[1:], over_vals, width = width, color = 'y', alpha = 0.5)
			
		if fig_file != None:
			plt.savefig(fig_file)
		if show:
			plt.show()
		
		
		
	
	@staticmethod
	def train(data, dist_choices, FORCE_APPART = False, bc_transform = True):
		"""
		Given a vector of data and list of distribution types the trainer
		will find the best fit for the mixture distribution.
		"""
		
		param_vec = N.array([])
		bounds = []
		
		if bc_transform:
			t_data, llm = boxcox(data)
		else:
			t_data = data
		
		#make initial guesses based on kmeans
		num_clust = len(dist_choices)
		init_weight = float(1)/float(num_clust)
		if num_clust == 1:
			centroids = [N.mean(t_data)]
			
		elif (num_clust == 2) & FORCE_APPART:
			c1 = N.max(t_data)
			c2 = N.min(t_data)
			
			centroids = N.array([c1, c2])
		else:
			(centroids, distortion) = kmeans(t_data, num_clust)
		centroids.sort()
		
		min_val = N.min(t_data)
		max_val = N.max(t_data)
		
		#create an "emperical" pdf
		bin_divisor = get_bin_divisor(len(t_data))

		n, bins = N.histogram(t_data, new=True,
				      bins = len(t_data)/bin_divisor, normed = True)
		if len(dist_choices) > 1:
			max_width = (bins[1]-bins[0])*bin_divisor
		else:
			max_width = None
		#create param_vec and bounds vector
		for this_dist, cent in izip(dist_choices, centroids):
			if this_dist == 'norm':
				try:
					param_vec = N.concatenate((param_vec, 
								   N.array([cent,1,init_weight])))
				except:
					
					param_vec = N.concatenate((param_vec, 
								   N.array([cent[0],1,init_weight])))
				bounds += [(min_val, max_val), (0, max_width), (0,1)]
			elif this_dist == 'uniform':
				param_vec = N.concatenate((param_vec, 
							   N.array([min_val,max_val,init_weight])))
				bounds += [(min_val, max_val), (min_val, max_val), (0,1)]
			elif this_dist == 'skewnorm':
				try:
					param_vec = N.concatenate((param_vec, 
								   N.array([cent,1,1,init_weight])))
				except:
					
					param_vec = N.concatenate((param_vec, 
								   N.array([cent[0],1,1,init_weight])))
				bounds += [(min_val, max_val), (0, max_width), (0, max_width), 
							(0,1)]
			else:
				raise KeyError, 'Unknown distribution %s' % this_dist
		#do the actual training
		
		param_val, like, d = fmin_tnc(score, param_vec, 
					      args = (dist_choices, bins[1:], n), 
					      approx_grad = True, bounds = bounds,
					      messages = 0)
		
		#make the trained distribution
		t_dist = m_modal()
		t_dist.use_bc = bc_transform
		t_dist.dists, t_w = unpack(param_val, dist_choices)
		
		#save normalized weights
		t_dist.weights = N.array(t_w)/N.array(t_w).sum()
		#TF, pval = t_dist.pval(data)
		
		return t_dist

def unpack(values, dist_choice):
	"""
	Unpacks the values based on the dist_choices provided.  The function
	returns a list of distributions and weights.
	"""
	
	val_list = values.tolist()
	
	dist_list = []
	w_list = []
	for ch in dist_choice:
		if ch == 'norm':
			dist_list.append(norm(val_list.pop(0), val_list.pop(0)))
		elif ch == 'uniform':
			dist_list.append(uniform(val_list.pop(0), val_list.pop(0)))
		elif ch == 'skewnorm':
			dist_list.append(skewnorm(val_list.pop(0), val_list.pop(0), 
						  val_list.pop(0)))
		else:
			raise KeyError, 'Unknown distribution %s' % ch
		w_list.append(val_list.pop(0))
	return dist_list, w_list
	


def score(in_vals, dist_choices, bin_pos, bin_pdf):
	
	d_list, w_list = unpack(in_vals, dist_choices)
	w_list = N.array(w_list)
	normed_w = w_list/w_list.sum()
	new_pdf = N.zeros_like(bin_pos)
	for dist, w in izip(d_list, w_list):
		new_pdf += dist.pdf(bin_pos) * w
	
	dist = N.sqrt(N.sum(N.power((bin_pdf - new_pdf),2)))
	return dist

	
def GatherData(HANDLE_OBJ, COL = None, SKIP_JUNK = True, SEP = '\t'):
	"""
	Gathers the data from an iterator.  The file can either be 1-column of
	numbers or a tab-delimeted file.  The program will by-default find the 
	first "float-able" column and use that as the values.  To override this 
	behaviour provide a COL kwarg.
	
	If SKIP_JUNK is true then the program will skip-over any lines which cannot
	be converted into floats ... if set to False then it will raise ValueError.
	"""
	out_data = []
	
	if COL == None:
		f_line = HANDLE_OBJ.next()
		f_parts = f_line.strip().split(SEP)
		if len(f_parts) == 0:
			out_data.append(float(f_parts[0]))
			COL = 0
		else:
			val = None
			for ind, part in izip(range(len(f_parts)), f_parts):
				try:
					val = float(part)
					break
				except ValueError:
					continue
			if val != None:
				out_data.append(val)
				COL = ind
			else:
				raise ValueError, 'Could not find float in: %s' % f_line
				
	for line in HANDLE_OBJ:
		parts = line.strip().split(SEP)
		try:
			val = float(parts[COL])
			out_data.append(val)
		except ValueError, IndexError:
			if SKIP_JUNK:
				continue
			else:
				raise ValueError, 'COL %d was not float in: %s' % (COL, line)
	return N.array(out_data)
	

def get_ratio(data):
	bm = m_modal.train(data, ['norm', 'norm'],  
			   bc_transform = False)
	um = m_modal([('norm', (data.mean(), data.std()), 1)])
	um.use_bc = False
	ll_2 = bm.log_likelihood(data)
	ratio = bm.get_ratio(data)
	return ratio

	
def go_file(fname):
	names = []
	vals = []
	with open(fname) as f:
		for line in f:
			[name, val] = line.strip().split('\t')
			names.append(name)
			vals.append(float(val))
	data = N.array(vals)
	real_ratio = get_ratio(data)

	m = data.mean()
	s = data.std()
	ratios = []
	samples = 100
	for x in xrange(samples):
		new_data_v = []
		for y in xrange(len(vals)):
			new_data_v.append(random.gauss(m, s))
		ratios.append(get_ratio(N.array(new_data_v)))
	random_is_better_eq = 0
	for r in ratios:
		if real_ratio <= r:
			random_is_better_eq += 1
#	go_file_old(fname)
	return [real_ratio, ratios, random_is_better_eq,
		str(random_is_better_eq) + '/'
		+ str(samples)]

def go_file_old(fname):
#	fname = sys.argv[1]
	names = []
	vals = []
	with open(fname) as f:
		for line in f:
			[name, val] = line.strip().split('\t')
			names.append(name)
			vals.append(float(val))
	data = N.array(vals)
	bm = m_modal.train(data, ['norm', 'norm'], FORCE_APPART=False,  
			   bc_transform = False)
	um = m_modal([('norm', (data.mean(), data.std()), 1)])
	um.use_bc = False
		#um,p = m_modal.train(data, ['norm'])
	
	#plt.subplot(211)
	bm.make_figure(data = data, plot_overlap = True)
	ll_2 = bm.log_likelihood(data)
	TF, p_val = bm.pval(data)
	plt.title('Bimodal Fit')
	xAxis = plt.axes().xaxis
	xAxis.set_label_text('Avg PCC')
	yAxis = plt.axes().yaxis
	yAxis.set_label_text('Density')
	#plt.subplot(212)
	#um.make_figure(data = data)
	ll_1 = um.log_likelihood(data)
	#plt.title('Unimodal Fit')
	
	fig_name = fname + '.png'
	plt.savefig(fig_name)
	plt.close()
	params = bm.GetParams()
	print bm.findcutoff(), params
	m1 = params[0][0]
	m2 = params[1][0]
	s1 = N.sqrt(params[0][1])
	s2 = N.sqrt(params[1][1])
	print m1, s1
	print m2, s2
	
	idx = bm.assign(data)
	vals0 = []
	names0 = []
	vals1 = []
	names1 = []
	for x in xrange(len(idx)):
		if idx[x] == 0:
			names0.append(names[x])
			vals0.append(vals[x])
		else:
			names1.append(names[x])
			vals1.append(vals[x])
	all_vals = [vals0, vals1]
	all_names = [names0, names1]
	data0 = N.array(vals0)
	data1 = N.array(vals1)
	if float(data0.mean()) < float(data1.mean()):
		files = ['date', 'party']
		print fname, files, data0.mean(), data1.mean()
	else:
		files = ['party', 'date']
		print fname, files, data1.mean(), data0.mean()
	for x in [0, 1]:
		with open(fname + '.' + files[x], 'w') as fnames:
			with open(fname + '.' + files[x] + '.vals', 'w') as fvals:
				for y in xrange(len(all_names[x])):
					fnames.write(all_names[x][y] + '\n')
					fvals.write(all_names[x][y] + '\t'
						    + str(all_vals[x][y]) + '\n')			

if __name__ == '__main__':
	bin_divisor = int(sys.argv[2])
	go_file_old(sys.argv[1])

#[ratio, random_ratios, 
	# better_count, pval] = go_file(sys.argv[1])
	#print ratio, pval
