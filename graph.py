import pandas as pd
import networkx as nx
from tqdm import tqdm



def create_graph(path='./reverb_wikipedia_tuples-1.1.txt'):
	reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
	df = pd.read_csv(path, sep='\t', header=None)
	df.columns = reverb_columns_name
	df = df.dropna()
	df = df.drop_duplicates()
	KBG = nx.MultiGraph()
	# nodes = df['narg1'].to_list() + df['narg2'].to_list()
	for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Reading Graph ...'):
		KBG.add_nodes_from([(row['narg1'], {'alias':row['arg1']})])
		KBG.add_nodes_from([(row['narg2'], {'alias':row['arg2']})])
		KBG.add_edges_from([(row['narg1'], row['narg2'], 
								{'alias':row['rel'], 'csents':row['csents'], 'conf':row['conf'], 'ExID':row['ExID']})])
		
		
	print(len(list(KBG.nodes)))
	print('')
	print(len(set(list(KBG.nodes))))


if __name__=='__main__':
	create_graph()