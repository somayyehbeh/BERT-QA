from transformers import BertModel, BertTokenizer
import torch

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

if __name__=='__main__':
    random_input = torch.randint(3, 20000, (5, 15)).long()
    bert = BertModel.from_pretrained("bert-base-uncased")
    node_edge_detector = NodeEdgeDetector(bert, dropout=torch.tensor(0.5))
    output = node_edge_detector(random_input)
    print(output.size())
    print(output)