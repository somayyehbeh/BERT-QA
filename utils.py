from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.DEBUG)

def read_data(path='./bertified/'):
	tokens_matrix = np.load(os.path.join(path, 'tokenmat.npy'))
	nods_borders = np.load(os.path.join(path, 'entities.npy'))
	edgs_spans = np.load(os.path.join(path, 'relations.npy'))
	borders = np.concatenate((nods_borders, edgs_spans), axis=1)
	X_train, X_test, y_train, y_test = train_test_split(tokens_matrix, borders, test_size=0.30, random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
	return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)		

def nodes_get_f1(predicts, golden):
	gold_start, gold_end = golden[:, 0], golden[:, 1]
	pred_start, pred_end = predicts[:, 0], predicts[:, 1]
	overlap_start = np.maximum(pred_start, gold_start)
	overlap_end = np.minimum(pred_end, gold_end)
	overlap = (overlap_end - overlap_start).sum()
	expected = (gold_end - gold_start).sum()
	predicted = (pred_end - pred_start).sum()
	recall = overlap / expected
	precision = overlap / predicted
	f1 = 2 * recall * precision / (recall + precision)
	logging.info("Dataset-wide F1, precision and recall:")
	logging.info(', '.join([str(item) for item in [f1.item(), precision.item(), recall.item()]]))
	overlap = (overlap_end - overlap_start)
	recall = overlap / (gold_end - gold_start)
	precision = overlap / (pred_end - pred_start)
	f1 = 2 * recall * precision / (recall + precision)
	recall = recall.mean()
	precision = precision.mean()
	acc = (f1 == 1).mean()
	f1 = f1.mean()
	logging.info("Averaged F1, precision and recall:")
	logging.info(', '.join([str(item) for item in [f1.item(), precision.item(), recall.item()]]))
	logging.info("Span accuracy")
	logging.info(acc)

def edges_get_f1(predicts, golden):
	rows, columns = golden.shape
	preds, actual, common, acc = 0, 0, 0, 0
	for question in range(rows):
		if np.array_equal(predicts[question], golden[question]):
			acc+=1
		for token in range(columns):
			if predicts[question, token]==1:
				preds+=1
			if golden[question, token]==1:
				actual+=1
			if golden[question, token]==1:
				if predicts[question, token]==1:
					common+=1
	precision = common/preds
	recall = common/actual
	f1 = 2 * recall * precision / (recall + precision)
	logging.info("Averaged F1, precision and recall:")
	logging.info(', '.join([str(item) for item in [f1, precision, recall]]))
	logging.info("Span accuracy")
	logging.info(acc/rows)


def get_tf_idf_query_similarity(vectorizer, docs_tfidf, query):
    """
    vectorizer: TfIdfVectorizer model
    docs_tfidf: tfidf vectors for all docs
    query: query doc

    return: cosine similarity between query and all docs
    """
    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    return cosineSimilarities

def get_hit(actual, predict):
	hit_1, hit_3, hit_5, hit_10, hit_100 = 0, 0, 0, 0, 0
	for index, item in enumerate(actual):
		for hit, prd in enumerate(predict[index]):
			if item==prd[0]:
				if hit<=0:
					hit_1+=1; hit_3+=1; hit_5+=1; hit_10+=1; hit_100+=1
					continue
				if hit<=3:
					hit_3+=1; hit_5+=1; hit_10+=1; hit_100+=1
					continue
				if hit<=5:
					hit_5+=1; hit_10+=1; hit_100+=1
					continue
				if hit<=10:
					hit_10+=1; hit_100+=1
					continue
				if hit<=100:
					hit_100+=1
					continue
	length = len(actual)
	hit_1/=length; hit_3/=length; hit_5/=length; hit_10/=length; hit_100/=length
	return hit_1, hit_3, hit_5, hit_10, hit_100
				 
if __name__=='__main__':
	train, valid, test = read_data()
	print(train[0].shape, train[1].shape)
	# get_f1(valid[1][:, :2], valid[1][:, :2])
	# print(train[0].shape)
	# print(get_hit([1, 2, 3], [[(1, 'test')], [(1, 'test')], [(3, 'test')]]))