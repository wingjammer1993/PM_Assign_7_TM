import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import math
from collections import OrderedDict

words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']


# This function creates synthetic documents
def create_documents(i_alpha, i_beta):
	documents = []
	word_topic_dist = np.random.dirichlet(alpha=[i_beta]*20, size=3)
	topic_draw = np.random.dirichlet([i_alpha]*3, size=200)
	for i in topic_draw:
		actual_doc = ''
		for j in range(0, 50):
			topic_distribution = np.random.multinomial(1, i, size=1)
			topic = np.argmax(topic_distribution)
			word_picked = np.random.multinomial(1, word_topic_dist[topic], size=1)
			word = int(np.argmax(word_picked))
			actual_doc = actual_doc + chr(65 + word) + " "
		documents.append(actual_doc)
	print(documents)
	return documents, word_topic_dist


# Function to display topics recovered by LDA model
def display_topics(model, feature_names, no_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print("Topic %d:" % topic_idx)
		print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def format_lda_p(lda_matrix, feature_names):
	lda_pr = np.zeros((3, 20))
	for i, topic in enumerate(lda_matrix):
		for j, word in enumerate(feature_names):
			true_idx = words.index(word)
			lda_pr[i][true_idx] = topic[j]
	return lda_pr


# In this function documents are count vectorized and passed to LDA
# The scores given by LDA are normalized to obtain P(W|T)
def latent_dirichlet_allocation(documents, i_num_topics, i_alpha, i_beta):
	count_vec = CountVectorizer(stop_words=None, analyzer='char', lowercase=False, max_df=0.99)
	tf = count_vec.fit_transform(documents)
	tf_feature_names = count_vec.get_feature_names()
	lda = LatentDirichletAllocation(n_components=i_num_topics, doc_topic_prior=i_alpha, topic_word_prior=i_beta).fit(tf)
	no_top_words = 10
	display_topics(lda, tf_feature_names, no_top_words)
	lda_prob = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
	lda_prob = format_lda_p(lda_prob, tf_feature_names)
	return lda_prob


# Plot the distribution of true and recovered topics
def plot_distributions(true_dist, recovered_dist):
	counter = 1
	x = np.arange(20)
	for i in range(0, 3):
		plt.subplot(2, 3, counter)
		plt.xticks(x, words)
		plt.xlabel('words')
		plt.ylabel('P(W|T)')
		plt.plot(true_dist[i], color='red')
		counter += 3
		plt.subplot(2, 3, counter)
		plt.xticks(x, words)
		plt.xlabel('words')
		plt.ylabel('P(W|T)')
		counter -= 2
		plt.plot(recovered_dist[i], color='green')
	plt.suptitle('Word-Topic distribution for all the topics')
	plt.show()


# Find the mean entropy, accepts a list of list as input
def find_mean_entropy(topic_doc):
	mean_entropy = 0
	for i in topic_doc:
		term = 0
		for j in i:
			if j != 0:
				term = term + j*math.log(j)
		mean_entropy = mean_entropy + term
	mean_entropy = -mean_entropy/len(topic_doc)
	return mean_entropy


# This function gives the variation of alpha vs. mean entropy for the recovered model, keeping beta constant
# True model has alpha = 0.1 and beta = 0.01
def get_alpha_distribution(alphas):
	mean_list = OrderedDict()
	# Find mean entropy of true model for reference
	docs_new, topic_draw = create_documents(0.1, 0.01)
	mean = find_mean_entropy(topic_draw)
	i_beta = 0.01
	# Find mean entropy of recovered model by varying alpha
	for alp in alphas:
		topic_list = []
		p_lda = latent_dirichlet_allocation(docs_new, 3, alp, i_beta)
		for i in docs_new:
			doc_topic = [0]*3
			for j in i:
				if j != " ":
					k = words.index(j)
					recovered_topic = np.argmax([p_lda[0][k], p_lda[1][k], p_lda[2][k]])
					doc_topic[int(recovered_topic)] += 1
			doc_topic = [x/50 for x in doc_topic]
			topic_list.append(doc_topic)
		mean_list[alp] = find_mean_entropy(topic_list)
	plt.title("Mean entropy for generative model is {}".format(mean))
	plt.xlabel("Alpha values")
	plt.ylabel("Mean Entropy of recovered model")
	plt.plot(list(mean_list.keys()), list(mean_list.values()))
	plt.show()


# This function gives the variation of beta vs. mean entropy for the recovered model, keeping alpha constant
# True model has beta = 0.01 and alpha = 0.1
def get_beta_distribution(betas):
	mean_list = OrderedDict()
	list_dummy = []
	i_alpha = 0.1
	# Find mean entropy of true model for reference
	docs_new, _ = create_documents(i_alpha, 0.01)
	ttd = get_word_topic_distribution(docs_new)
	for i in ttd:
		list_dummy.append(ttd[i])
	mean = find_mean_entropy(list_dummy)
	# Find mean entropy of recovered model by varying beta
	for bet in betas:
		list_dummy = []
		p_lda = latent_dirichlet_allocation(docs_new, 3, i_alpha, bet)
		for i in p_lda:
			list_dummy.append(i)
		mean_list[bet] = find_mean_entropy(list_dummy)
	plt.title("Mean entropy for generative model is {}".format(mean))
	plt.xlabel("Beta values")
	plt.ylabel("Mean Entropy of recovered model")
	plt.plot(list(mean_list.keys()), list(mean_list.values()))
	plt.show()


if __name__ == "__main__":
	# Part 1
	alpha = 0.1
	beta = 0.01
	num_topics = 3
	docs, w_given_t = create_documents(alpha, beta)
	print(docs[0])

	# Part 2
	lda_p = latent_dirichlet_allocation(docs, num_topics, alpha, beta)
	plot_distributions(w_given_t, lda_p)

	# Part 3
	list_alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
	#get_alpha_distribution(list_alpha)

	list_beta = [0.01, 0.03, 0.05, 0.07, 0.09]
	#get_beta_distribution(list_beta)
	print('done')




