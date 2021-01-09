from transformers import BertModel, BertTokenizer
from transformers import AdamW
import torch
from torch.nn.functional import nll_loss
from torch.utils.data import Dataset, DataLoader
from utils import read_data, get_f1
from torch.nn.functional import mse_loss
from torch.nn.functional import one_hot
import numpy as np 
from tabulate import tabulate
from tqdm import tqdm



import logging
logging.basicConfig(level=logging.DEBUG)



class NodeEdgeDetector(torch.nn.Module):
	'''
	Neural Network architecture!
	'''
	def __init__(self, bert, tokenizer, dropout=0.5, clip_len=True, **kw):
		super().__init__(**kw)
		self.bert = bert
		dim = self.bert.config.hidden_size
		self.nodestart = torch.nn.Linear(dim, 1)
		self.nodeend = torch.nn.Linear(dim, 1)
		
		self.edgestart = torch.nn.Linear(dim, 1)
		self.edgeend = torch.nn.Linear(dim, 1)
		self.dropout = torch.nn.Dropout(p=dropout)
		self.clip_len = clip_len

		self.tokenizer = tokenizer

	def forward(self, x):	   # x: (batsize, seqlen) ints
		mask = (x != 0).long()
		if self.clip_len:
			maxlen = mask.sum(1).max().item()
			maxlen = min(x.size(1), maxlen + 1)
			mask = mask[:, :maxlen]
			x = x[:, :maxlen]
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
	'''
		Convert Data to proper Tensor dataset
	'''
	def __init__(self, data):
		# convert into tensors
		self.tokens_matrix = torch.from_numpy(data[0]).long()
		self.borders = torch.from_numpy(data[1]).long()
		self.n_samples = data[0].shape[0]

	def __getitem__(self, index):
		# returns specific item
		return self.tokens_matrix[index], self.borders[index] 
	def __len__(self):
		return self.n_samples
		# returns dataset length



class TrainingLoop:
	'''
	Everything related to model training
	'''
	def __init__(self, model, optimizer, freezeemb=True, 
				 epochs=6, save_path='./models/', **kw):
		self.model = model
		params = []
		for paramname, param in self.model.named_parameters():
			if paramname.startswith("bert.embeddings.word_embeddings"):
				if not freezeemb:
					params.append(param)
			else:
				params.append(param)
		self.optimizer = optimizer(params, **kw)
		self.epochs = epochs
		self.save_path = save_path
		self.predicts = None
	def train(self, dataloader, eval_dataloader, loss_function):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model.to(device)
		
		for epoch in range(self.epochs):
			self.model.train()
			losses = []

			for _, batch in enumerate(tqdm(dataloader, desc=f"Train Epoch Number {epoch+1}")):
				self.model.zero_grad()
				X, y = batch
				X = X.to(device); y = y.to(device)
				logits = self.model(X) 
				loss = loss_function(logits, one_hot(y, num_classes=logits.size()[-1]).float(),
							 reduction='sum')
				losses.append(loss)
				loss.backward()
				self.optimizer.step()
			logging.info(f'Epoch number: {epoch+1} Train Loss is equal: {sum(losses)/len(losses)}') 
			self.eval(eval_dataloader, loss_function, epoch, device)


	def eval(self, dataloader, loss_function, epoch, device):
		self.model.eval()
		losses = []
		for _, batch in enumerate(tqdm(dataloader, desc=f"Eval Epoch Number {epoch+1}")):
			with torch.no_grad():
				X, y = batch
				X = X.to(device); y = y.to(device)
				logits = self.model(X)
				loss = loss_function(logits, one_hot(y, num_classes=logits.size()[-1]).float(),
							 reduction='sum')
				losses.append(loss)
		logging.info(f'Epoch number: {epoch+1} Eval Loss is equal: {sum(losses)/len(losses)}')

	def predict(self, dataloader, device, evaluate=True):
		self.model.eval()
		predicts = []
		for _, batch in enumerate(tqdm(dataloader, desc=f"Predicting ...")):
			with torch.no_grad():
				X, _ = batch
				X = X.to(device)
				logits = self.model(X)
				borders = torch.argmax(logits, dim=2).cpu().detach().numpy().tolist()
				[predicts.append(item) for item in borders]
		predicts = np.array(predicts)
		self.predicts = predicts
		if evaluate:
			goldens = []
			for batch in dataloader:
				_, y = batch
				borders = y.cpu().detach().numpy().tolist()
				[goldens.append(item) for item in borders]
			goldens = np.array(goldens)
			gold_nodes_border = goldens[:, :2]
			gold_edges_border = goldens[:, 2:]
			pred_nodes_border = self.predicts[:, :2]
			pred_edges_border = self.predicts[:, 2:]
			get_f1(pred_nodes_border, gold_nodes_border)
			get_f1(pred_edges_border, gold_edges_border)
	
	def save(self, save_path='./models/node_edge_bert.pt'):
		torch.save(self.model, save_path)
	
	def load(self, save_path='./models/node_edge_bert.pt'):
		self.model = torch.load(save_path)

	def readable_predict(self, _input='Where was Bill Gates Born?'):
		addspecialtokens = lambda string:f'[CLS] {string} [SEP]'
		wordstoberttokens = lambda string:self.model.tokenizer.tokenize(string)
		berttokenstoids = lambda tokens:self.model.tokenizer.convert_tokens_to_ids(tokens)
		input_token_ids = berttokenstoids(wordstoberttokens(addspecialtokens(_input)))
		input_tensors = torch.tensor([input_token_ids]).long()
		with torch.no_grad():
			_output = self.model(input_tensors)
		borders = torch.argmax(_output, dim=2).detach().cpu().numpy()[0]
		node = self.model.tokenizer.convert_ids_to_tokens(input_token_ids[borders[0]:borders[1]])
		edge = self.model.tokenizer.convert_ids_to_tokens(input_token_ids[borders[2]:borders[3]])
		data = [[_input, node, edge]]
		print(tabulate(data, headers=["Question", "Node", "Edge"]))
		



		




	  







if __name__=='__main__':
	train, valid, test = read_data()
	bert = BertModel.from_pretrained("bert-base-uncased")
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	node_edge_detector = NodeEdgeDetector(bert, tokenizer, dropout=torch.tensor(0.5))
	optimizer = AdamW
	kw = {'lr':0.0002, 'weight_decay':0.1}
	tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
	
	train_dataset = BordersDataset(train)
	train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, pin_memory=True)
	valid_dataset = BordersDataset(valid)
	valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=4, shuffle=False, pin_memory=True)
	test_dataset = BordersDataset(test)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, pin_memory=True)
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	loss = mse_loss
	tl.train(train_dataloader, valid_dataloader, loss)
	tl.save()
	##################################################
	tl.load()
	tl.predict(test_dataloader, device)
	##################################################
	tl.readable_predict()
	