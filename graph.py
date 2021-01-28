import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import get_tf_idf_query_similarity

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
		self.nodes_vectorizer = TfidfVectorizer()
		self.edges_vectorizer = TfidfVectorizer()
		self.nodes_tfidf = self.nodes_vectorizer.fit_transform(self.nodes)
		self.edges_tfidf = self.edges_vectorizer.fit_transform(self.edges.values())


	def nodesquery(self, search_phrase, cutoff=80):
		candidates = process.extractBests(search_phrase, self.nodes, score_cutoff=cutoff, limit=len(self.nodes)-1)
		return candidates

	def edgesquery(self, search_phrase, cutoff=80):
		candidates = process.extractBests(search_phrase, self.edges, score_cutoff=cutoff, limit=len(self.edges)-1)
		return candidates

	def query(self, node='Bill Gates', edge='Born'):
		nodes = self.nodesquery(node)
		print(nodes)
		edges = self.edgesquery(edge)
		print(edges)
		candidates = []
		for nd in nodes:
			for ed in edges:
				if ed[-1][0]==nd[-1]:
					candidates.append((nd[0]['reverb_line_no'], nd[-1], nd[1], ed[0], ed[1], ed[-1][1]))
				if ed[-1][1]==nd[-1]:
					candidates.append((nd[0]['reverb_line_no'], nd[-1], nd[1], ed[0], ed[1], ed[-1][0]))

		return candidates

	def tfidf_nodes_query(self, search_phrase, cutoff=80):
		similarities = get_tf_idf_query_similarity(self.nodes_vectorizer, self.nodes_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.nodes, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)}

		return sorted_ranks

	def tfidf_edges_query(self, search_phrase, cutoff=80):
		similarities = get_tf_idf_query_similarity(self.edges_vectorizer, self.edges_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.edges.keys(), similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)}
		return sorted_ranks

	def tfidf_query(self, node='Bill Gates', edge='Born'):
		nodes = self.nodesquery(node)
		print(nodes)
		edges = self.edgesquery(edge)

class ReverbKnowledgeBase:
	def __init__(self, path='./reverb_wikipedia_tuples-1.1.txt'):
		super().__init__()
		df = pd.read_csv(path, sep='\t', header=None)
		reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
		df.columns = reverb_columns_name
		df = df.dropna()
		df = df.drop_duplicates()
		self.KB = df
		self.nodes = self.KB['arg1'].to_list()+self.KB['arg2'].to_list()
		self.edges = self.KB['rel'].to_list()
		self.nodes_vectorizer = TfidfVectorizer()
		self.edges_vectorizer = TfidfVectorizer()
		self.nodes_tfidf = self.nodes_vectorizer.fit_transform(self.nodes)
		self.edges_tfidf = self.edges_vectorizer.fit_transform(self.edges)
		self.relations = {}
		for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Indexing ...'):
			if row['rel'] in self.relations:
				self.relations[row['rel']].append((row['arg1'], index, row['conf']))
				self.relations[row['rel']].append((row['arg2'], index, row['conf']))
			else:
				self.relations[row['rel']] = [(row['arg1'], index, row['conf'])]
				self.relations[row['rel']].append((row['arg2'], index, row['conf']))
			
	def tfidf_nodes_query(self, search_phrase, cutoff=50):
		similarities = get_tf_idf_query_similarity(self.nodes_vectorizer, self.nodes_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.nodes, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)[:min(len(ranks), cutoff)]}

		return sorted_ranks

	def tfidf_edges_query(self, search_phrase, cutoff=50):
		similarities = get_tf_idf_query_similarity(self.edges_vectorizer, self.edges_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.edges, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)[:min(len(ranks), cutoff)]}
		return sorted_ranks

	def tfidf_query(self, node='Bill Gates', edge='Born'):
		nodes = self.tfidf_nodes_query(node)
		edges = self.tfidf_edges_query(edge)
		pruned = []
		for node in nodes.keys():
			for edge in edges.keys():
				for item in self.relations[edge]:
					if item[0]==node:
						pruned.append((item[1], item[-1], nodes[node], edges[edge]))
		sorted_pruned = sorted(pruned, key=lambda x:x[2]+x[3], reverse=True)
		return sorted_pruned
		

if __name__=='__main__':
	RKBG = ReverbKnowledgeBase() #	'./sample_reverb_tuples.txt'
	# print(RKBG.edges)
	# print(RKBG.tfidf_nodes_query('fishkind'))
	# print(RKBG.tfidf_edges_query('grew up in'))
	print(RKBG.tfidf_query(node='fishkind', edge='grew up in'))

	# RKBG = ReverbKnowledgeBaseGraph() #	'./sample_reverb_tuples.txt'
	# # print(RKBG.edges)
	# print(RKBG.tfidf_nodes_query('fishkind'))
	# print(RKBG.tfidf_edges_query('grew up in'))
	# print('candidates')
	# fishkind, Edge:did fishkind grow up
	# print(RKBG.query(node='fishkind', edge='grew up in'))