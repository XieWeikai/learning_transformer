import torch
from torch import nn
from torch import optim
from torch.nn import Transformer
from torch.nn.functional import log_softmax
from torch.nn.functional import kl_div

# Hyperparameters
vocab_size = 101
embedding_size = 64
nbatches = 100
batch_size = 40
sentence_len = 100
log_step = 20
lr = 0.001
smoothing = 0.2
epoches = 20
N = 2

class CopyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.src_embedding = nn.Embedding(vocab_size, embedding_size)
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = Transformer(embedding_size, nhead=8, num_encoder_layers=6, num_decoder_layers=6, batch_first=True)
        self.generator = nn.Linear(embedding_size, vocab_size)

    def encode(self, src, src_mask=None):
        src = self.src_embedding(src)
        memory = self.transformer.encoder(src, src_mask)
        return memory
    
    def decode(self, memory, tgt, tgt_mask=None):
        tgt = self.tgt_embedding(tgt)
        output = self.transformer.decoder(tgt, memory, tgt_mask)
        return output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_embed = self.src_embedding(src)
        tgt_embed = self.tgt_embedding(tgt)

        transformer_output = self.transformer(src_embed, tgt_embed, src_mask, tgt_mask)

        output = self.generator(transformer_output)
        output = log_softmax(output, dim=-1)  # Apply log_softmax

        return output
    
# def data_generator(nbatches, batch_size, vocab_size, max_sentence_len):
#     for _ in range(nbatches):
#         src_batch, tgt_batch = [], []

#         for _ in range(batch_size):
#             sentence_len = torch.randint(2, max_sentence_len, size=(1,)).item()

#             # Generate random sentences
#             src = torch.randint(3, vocab_size, size=(sentence_len - 2,))
#             src = torch.cat([torch.full((1,), 1), src, torch.full((1,), 2)])

#             # The target is the same as the source for the copy task
#             tgt = src.clone()

#             src_batch.append(src)
#             tgt_batch.append(tgt)

#         # Pad sequences with 0s to the maximum sentence length
#         src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
#         tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)

#         yield src_batch, tgt_batch
        
def data_generator(nbatches, batch_size, vocab_size, sentence_len): # without padding
    for _ in range(nbatches):
        # Generate random sentences
        src = torch.randint(3, vocab_size, size=(batch_size, sentence_len - 2))
        src = torch.cat([torch.full((batch_size, 1), 1), src, torch.full((batch_size, 1), 2)], dim=1)

        # The target is the same as the source for the copy task
        tgt = src.clone()

        yield src, tgt

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.2, ignore_index=0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index
        self.dim = dim

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.ignore_index] = 0
            mask = torch.nonzero(target.data == self.ignore_index)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return kl_div(pred, true_dist, reduction='sum')

def train_copy_model():
    # Create the model, loss function and optimizer
    model = CopyTransformer(vocab_size, embedding_size, num_encoder_layers=N, num_decoder_layers=N)
    model.train()
    loss_fn = LabelSmoothingLoss(vocab_size, smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epoches):
        print(f"epoch: {epoch + 1}")
        for i, (src, tgt) in enumerate(data_generator(nbatches, batch_size, vocab_size, sentence_len)):
            optimizer.zero_grad()

            # Forward pass
            output = model(src, tgt[:,:-1])

            # Compute loss
            loss = loss_fn(output.reshape(-1, vocab_size), tgt[:,1:].reshape(-1))
            tokens = tgt[:,1:].numel()
            loss /= tokens

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Logging
            if i % log_step == 0:
                print(f'Batch {i}, Loss {loss.item()}')
    
    return model

def greedy_inference(model, src, max_len, start_symbol):
    model.eval()
    
    memory = model.encode(src)

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, ys)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

import torch.nn.functional as F

def sample_inference(model, src, max_len, start_symbol, topk=None):
    model.eval()
    
    memory = model.encode(src)

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, ys)
        linear_output = model.generator(out[:, -1]) # (b, vocab_size)
        prob = F.softmax(linear_output, dim=-1)   # (b, vocab_size)
        if topk is not None:
            prob = select_topk(prob, topk)
        next_word = torch.multinomial(prob, num_samples=1) # (b, 1)
        ys = torch.cat([ys, 
                        next_word], dim=1)
    return ys

def select_topk(dist, k):
    # dist: (b,classes)
    _, indices = torch.topk(dist, k=dist.size(-1)-k, dim=-1,largest=False) # 挑较小的 classes - k项
    dist.scatter_(1, indices, 0) # 将这些项置为0
    return dist / dist.sum(dim=-1, keepdim=True) # 让概率总和为1    

from_file = True

if __name__ == "__main__":
    # To use it
    copy_model = None
    if from_file:
        try:
            copy_model = CopyTransformer(vocab_size, embedding_size, num_encoder_layers=N, num_decoder_layers=N)
            copy_model.load_state_dict(torch.load('torch_copy_model.pt'))
            copy_model.eval()
            print("Model loaded from torch_copy_model.pt")
        except FileNotFoundError:
            copy_model = None
            print("torch_copy_model.pt not found, training a new model...")
            
    if copy_model is None:
        copy_model = train_copy_model()
        # Save the model
        torch.save(copy_model.state_dict(), 'torch_copy_model.pt')
        print("Model saved to torch_copy_model.pt")
    
    src = torch.randint(1,20,(1,10)).long()
    print(f"src: {src}")
    output = greedy_inference(copy_model, src, max_len=20, start_symbol=1)
    print(f'greedy_inference: {output}')
    output = sample_inference(copy_model, src, max_len=20, start_symbol=1, topk=5)
    print(f'sample_inference: {output}')
    
