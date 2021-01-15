import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm


class ReverbKnowledgeBaseGraph:
	def __init__(self, path='./reverb_wikipedia_tuples-1.1.txt'):
		super().__init__()
		df = pd.read_csv(path, sep='\t', header=None)
		reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
		df.columns = reverb_columns_name
		df = df.dropna()
		df = df.drop_duplicates()
		KBG = nx.MultiGraph()
		for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Reading Graph ...'):
			KBG.add_nodes_from([(row['arg1'], {'alias':row['narg1'], 'reverb_line_no':index})])
			KBG.add_nodes_from([(row['arg2'], {'alias':row['narg2'], 'reverb_line_no':index})])
			KBG.add_edges_from([(row['arg1'], row['arg2'], 
									{'nrel':row['nrel'], 'alias':row['rel'], 'csents':row['csents'], 'conf':row['conf'], 'ExID':row['ExID']})])
		self.KBG = KBG
		self.nodes = self.KBG.nodes
		self.edges = nx.get_edge_attributes(self.KBG,'alias')

	def nodesquery(self, search_phrase, limit=50):
		candidates = process.extract(search_phrase, self.nodes, limit=limit)
		return candidates

	def edgesquery(self, search_phrase, limit=250):
		candidates = process.extract(search_phrase, self.edges, limit=limit)
		return candidates

	def query(self, node='Bill Gates', edge='Born'):
		nodes = self.nodesquery(node)
		edges = self.edgesquery(edge)
		candidates = []
		for nd in nodes:
			for ed in edges:
				if ed[-1][0]==nd[-1]:
					candidates.append((nd[0]['reverb_line_no'], nd[-1], nd[1], ed[0], ed[1], ed[-1][1]))
				if ed[-1][1]==nd[-1]:
					candidates.append((nd[0]['reverb_line_no'], nd[-1], nd[1], ed[0], ed[1], ed[-1][0]))

		return candidates




if __name__=='__main__':
	RKBG = ReverbKnowledgeBaseGraph()
	print('candidates')
	print(RKBG.query(node='Lenin', edge='fled to'))