import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def create_documents(i_alpha, i_beta):
	documents = []
	topic_draw = np.random.dirichlet([i_alpha]*3, size=200)
	for i in topic_draw:
		actual_doc = ''
		for j in range(0, 50):
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
	return documents


def latent_dirichlet_allocation(documents, i_num_topics, i_alpha, i_beta):
	count_vec = CountVectorizer(stop_words=None, analyzer='char', lowercase=False, max_df=0.99)
	tf = count_vec.fit_transform(documents)
	tf_feature_names = count_vec.get_feature_names()
	lda = LatentDirichletAllocation(n_components=i_num_topics, doc_topic_prior=i_alpha, topic_word_prior=i_beta).fit(tf)
	no_top_words = 10
	display_topics(lda, tf_feature_names, no_top_words)
	lda_prob = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
	return lda_prob


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



if __name__ == "__main__":
	alpha = 0.1
	beta = 0.01
	num_topics = 3
	docs = create_documents(alpha, beta)
	lda_p = latent_dirichlet_allocation(docs, num_topics, alpha, beta)
	ttd_p = get_word_topic_distribution(docs)
	plot_distributions(ttd_p, lda_p)
	print('done')





