from node_edge_bert import *
from graph import *
import pandas as pd

if __name__=='__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	bert = BertModel.from_pretrained("bert-base-uncased")
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	node_edge_detector = NodeEdgeDetector(bert, tokenizer, dropout=torch.tensor(0.5))
	optimizer = AdamW
	kw = {'lr':0.0002, 'weight_decay':0.1}
	tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
	loss = mse_loss
	tl.load()

	RKBG = ReverbKnowledgeBaseGraph()
	print(RKBG.query(node='Lenin', edge='fled to'))

	test_df = pd.read_excel('./test.xlsx')
	system_results = []
	for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
		node, edge = tl.readable_predict(device, _input=row['Question'], print_result=False)
		node = ' '.join(node); edge = ' '.join(edge)
		node = node.replace(' ##', ''); edge = edge.replace(' ##', '')
		print(f'Node: {node}, Edge:{edge}')
		temp = RKBG.query(node=node, edge=edge)
		print(temp)
		system_results.append(temp)

		
