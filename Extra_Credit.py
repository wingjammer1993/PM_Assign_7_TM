from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import io


def big_doc_c(input_document):
    ip_test = []
    ip = io.open(input_document,'r', encoding='utf8')
    for ip_line in ip.readlines():
        result = ''.join([i for i in ip_line if not i.isdigit()])
        ip_test.append(result)
    return ip_test


def display_topics(model, feature_names, top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-top_words - 1:-1]]))


text = big_doc_c('medical.txt')
no_features = 1000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(text)
tf_feature_names = tf_vectorizer.get_feature_names()
no_topics = 10
# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)

