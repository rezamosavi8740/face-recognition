import torch.nn

class ModelWithEmbedding(nn.Module):
    def __init__(self, base_model, embedding_dim_input = 512 ,embedding_dim_output = 756):
        super().__init__()
        self.base_model = base_model
        self.embedding_layer = nn.Embedding(num_embeddings=embedding_dim_input, embedding_dim=embedding_dim_output)

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x)
        return x
