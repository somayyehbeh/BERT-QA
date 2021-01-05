from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np 

class NodeEdgeDetector(torch.nn.Module):
    def __init__(self, bert, dropout=0., **kw):
        super().__init__(**kw)
        self.bert = bert
        dim = self.bert.config.hidden_size
        self.nodestart = torch.nn.Linear(dim, 1)
        self.nodeend = torch.nn.Linear(dim, 1)
        
        self.edgestart = torch.nn.Linear(dim, 1)
        self.edgeend = torch.nn.Linear(dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):       # x: (batsize, seqlen) ints
        mask = (x != 0).long()
        bert_outputs = self.bert(x, attention_mask=mask, output_hidden_states=False)
        lhs = bert_outputs.last_hidden_state
        a = self.dropout(lhs)
        logits_node_start = self.nodestart(lhs)
        logits_node_end = self.nodeend(lhs)
        logits_edge_start = self.edgestart(lhs)
        logits_edge_end = self.edgeend(lhs)
        
        logits = torch.cat([logits_node_start.transpose(1, 2), logits_node_end.transpose(1, 2), 
                            logits_edge_start.transpose(1, 2), logits_edge_end.transpose(1, 2)], 1)
        return logits


class BordersDataset(Dataset):
	def __init__(self, path='./bertified/'):
		# loading datasets into the memory
		self.tokens_matrix = np.load(os.path.join(path, 'tokenmat.npy'))
		self.nods_borders = np.load(os.path.join(path, 'entities.npy'))
		self.edgs_borders = np.load(os.path.join(path, 'relations.npy'))
		# convert into tensors
		self.tokens_matrix = torch.from_numpy(self.tokens_matrix).long()
		self.borders = torch.from_numpy(np.concatenate((self.nods_borders, self.edgs_borders),
														 axis=1)).long()		
		self.n_samples = self.borders.shape[0]

	def __getitem__(self, index):
		# returns specific item
		return self.tokens_matrix[index], self.borders[index] 
	def __len__(self):
		return self.n_samples
		# returns dataset length



if __name__=='__main__':
    # dataset = BordersDataset()
    # dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
    # dataiter = iter(dataloader)
    # data = dataiter.next()
    # featurs, labels = data
    # print(featurs.size, labels.size, labels)
    # random_input = torch.randint(3, 20000, (5, 15)).long()
    # bert = BertModel.from_pretrained("bert-base-uncased")
    # node_edge_detector = NodeEdgeDetector(bert, dropout=torch.tensor(0.5))
    # output = node_edge_detector(random_input)
    # print(output.size())
    # print(output)
    # torch.save(node_edge_detector, './models/save_test.pt')
    # model = torch.load('./models/save_test.pt')
    # model.eval()
    # output = model(random_input)
    # print(output.size())
    # print(output)
    