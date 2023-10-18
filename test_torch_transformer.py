import torch

import torch_transformer as tt
from torch_transformer import CopyTransformer

def test_copy_transformer():
    # Define hyperparameters
    vocab_size = 100
    embedding_size = 32
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # Create a CopyTransformer instance
    model = CopyTransformer(vocab_size, embedding_size, nhead, num_encoder_layers, num_decoder_layers)
    model.eval()
    # Define input tensors
    src = torch.randint(low=0, high=vocab_size, size=(2, 10))
    tgt = torch.randint(low=0, high=vocab_size, size=(2, 12))

    # Test forward pass
    output = model(src, tgt)
    assert output.shape == (2, 12, vocab_size)

    # calc exp(output) because generator calc log_softmax
    output = torch.exp(output)
    
    # Test that output probabilities sum to 1
    assert torch.allclose(output.sum(dim=-1), torch.ones((2, 12)), rtol=1e-5)

    # Test that output probabilities are non-negative
    assert torch.all(output >= 0)

    # Test that the model can be saved and loaded
    torch.save(model.state_dict(), 'test_model.pt')
    loaded_model = CopyTransformer(vocab_size, embedding_size, nhead, num_encoder_layers, num_decoder_layers)
    loaded_model.load_state_dict(torch.load('test_model.pt'))
    loaded_model.eval()
    assert torch.allclose(model(src, tgt), loaded_model(src, tgt), rtol=1e-5)

    print("copy_transformer tests passed!")
    
def test_data_generator():
    # Define hyperparameters
    nbatches = 2
    batch_size = 3
    vocab_size = 100
    sentence_len = 10

    # Create a data generator instance
    data_gen = tt.data_generator(nbatches, batch_size, vocab_size, sentence_len)

    # Test that the generator produces the expected number of batches
    assert sum(1 for _ in data_gen) == nbatches

    # Reset the generator
    data_gen = tt.data_generator(nbatches, batch_size, vocab_size, sentence_len)

    # Test that each batch has the expected shape
    for src, tgt in data_gen:
        # print(src)
        # print(tgt)
        assert torch.allclose(src, tgt)

    print("data_generator tests passed!")
    
if __name__ == '__main__':
    test_copy_transformer()
    test_data_generator()