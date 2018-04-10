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
	# Draw the topics from Dirichlet with alphas as [0.1, 0.1, 0.1]
	topic_draw = np.random.dirichlet([i_alpha]*3, size=200)
	for i in topic_draw:
		actual_doc = ''
		# For each word in the document
		for j in range(0, 50):
			# Select a topic the particular word will belong to with multinomial using the topic_draws generated above
			# Choose the word in the topic which gives the maximum probability from dirichlet
			# It is assumed that topic 0 will have words with equal probability from A-G and 0 otherwise
			# Similarly H to N for topic 1 and O to T for topic 2
			topic_distribution = np.random.multinomial(1, i, size=1)
			if topic_distribution[0][0] == 1:
				topic_0 = np.argmax(np.random.dirichlet([i_beta]*7))
				actual_doc = actual_doc + (chr(65 + int(topic_0))) + " "  # A to G
			if topic_distribution[0][1] == 1:
				topic_1 = np.argmax(np.random.dirichlet([i_beta]*7))
				actual_doc = actual_doc + (chr(72 + int(topic_1))) + " "  # H to N
			if topic_distribution[0][2] == 1:
				topic_2 = np.argmax(np.random.dirichlet([i_beta]*6))
				actual_doc = actual_doc + (chr(79 + int(topic_2))) + " "  # O to T
		documents.append(actual_doc)
	print(documents)
	return documents, topic_draw


# Function to display topics recovered by LDA model
def display_topics(model, feature_names, no_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print("Topic %d:" % topic_idx)
		print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


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
	return lda_prob


# This function gives the P(W|T) of the generated documents
def get_word_topic_distribution(documents):
	topic_document = [0]*3
	topic_0 = [0]*20
	topic_1 = [0]*20
	topic_2 = [0]*20
	topics = {}
	for idx, i in enumerate(documents):
		for j in i:
			if j != " ":
				if 0 <= words.index(j) < 7:
					topic_document[0] = topic_document[0] + 1
					topic_0[words.index(j)] = topic_0[words.index(j)] + 1
				elif 7 <= words.index(j) < 14:
					topic_document[1] = topic_document[1] + 1
					topic_1[words.index(j)] = topic_1[words.index(j)] + 1
				else:
					topic_document[2] = topic_document[2] + 1
					topic_2[words.index(j)] = topic_2[words.index(j)] + 1
	topics[0] = [x/topic_document[0] for x in topic_0]
	topics[1] = [x/topic_document[1] for x in topic_1]
	topics[2] = [x/topic_document[2] for x in topic_2]
	return topics


# Plot the distribution of true and recovered topics
def plot_distributions(true_dist, recovered_dist):

	for i in range(0, 3):
		x = np.arange(20)
		plt.xticks(x, words)
		plt.xlabel('words')
		plt.ylabel('P(W|T)')
		plt.title('Word-Topic distribution for all the topics')
		plt.plot(true_dist[i], color='red')
		plt.plot(recovered_dist[i], color='green')
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
	# Find mean entropy of recovered model by varying beta
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
	# Vary beta for the recovered model
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
	docs, t_given_d = create_documents(alpha, beta)
	print(docs[0])

	# Part 2
	lda_p = latent_dirichlet_allocation(docs, num_topics, alpha, beta)
	ttd_p = get_word_topic_distribution(docs)
	plot_distributions(ttd_p, lda_p)

	# Part 3
	list_alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
	get_alpha_distribution(list_alpha)

	list_beta = [0.01, 0.03, 0.05, 0.07, 0.09]
	get_beta_distribution(list_beta)
	print('done')




