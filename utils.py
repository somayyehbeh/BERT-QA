from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging
logging.basicConfig(level=logging.DEBUG)

def read_data(path='./bertified/'):
	tokens_matrix = np.load(os.path.join(path, 'tokenmat.npy'))
	nods_borders = np.load(os.path.join(path, 'entities.npy'))
	edgs_borders = np.load(os.path.join(path, 'relations.npy'))
	borders = np.concatenate((nods_borders, edgs_borders), axis=1)
	X_train, X_test, y_train, y_test = train_test_split(tokens_matrix, borders, test_size=0.30, random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
	return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)		

def get_f1(predicts, golden):
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

if __name__=='__main__':
	train, valid, test = read_data()
	get_f1(valid[1][:, :2], valid[1][:, :2])
	print(train[0].shape)