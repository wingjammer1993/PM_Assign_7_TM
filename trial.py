import numpy as np
from math import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def create_documents(i_alpha, i_beta):
	words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T']
	documents = []
	for i in range(0, 200):
		doc = []
		actual_doc = []
		topic_draw = np.random.dirichlet([i_alpha]*3)
		samples_per_topic = [floor(x*50) for x in topic_draw]
		if sum(samples_per_topic) != 50:
			num = np.random.choice([0, 1, 2])
			samples_per_topic[num] = samples_per_topic[num] + (50 - sum(samples_per_topic))
			topic_0 = [floor(x*samples_per_topic[0]) for x in np.random.dirichlet([i_beta]*7)]  # A to G
			topic_1 = [floor(x*samples_per_topic[1]) for x in np.random.dirichlet([i_beta]*7)]  # H to N
			topic_2 = [floor(x*samples_per_topic[2]) for x in np.random.dirichlet([i_beta]*6)]  # O to T
			sum_topic = sum(topic_0) + sum(topic_1) + sum(topic_2)
			random_topic = np.random.choice(words, size=(50 - abs(sum_topic)))
			doc.extend(topic_0)
			doc.extend(topic_1)
			doc.extend(topic_2)
			for idx, j in enumerate(doc):
				letter = chr(65+idx)+" "
				num_letter = doc[idx]
				ls = [letter]*num_letter
				actual_doc.extend(ls)
			actual_doc.extend(random_topic)
		documents.append(actual_doc)
	return documents


def latent_dirichlet_allocation(documents, i_num_topics, i_alpha, i_beta):
	count_vec = CountVectorizer()
	tf = count_vec.fit_transform(documents)
	tf_feature_names = count_vec.get_feature_names()
	lda = LatentDirichletAllocation(n_topics=i_num_topics, doc_topic_prior=i_alpha, topic_word_prior=i_beta).fit(tf)
	no_top_words = 10
	display_topics(lda, tf_feature_names, no_top_words)


if __name__ == "__main__":
	alpha = 0.1
	beta = 0.01
	num_topics = 3
	docs = create_documents(alpha, beta)
	latent_dirichlet_allocation(docs, num_topics, alpha, beta)


