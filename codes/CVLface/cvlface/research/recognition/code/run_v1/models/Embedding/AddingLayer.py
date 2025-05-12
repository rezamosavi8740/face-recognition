import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConfig:
    def __init__(self, color_space='RGB', freeze=False, num_classes=None, embedding_size=512):
        self.color_space = color_space
        self.freeze = freeze  # Required by the pipeline
        self.num_classes = num_classes  # Required for classifier compatibility
        self.embedding_size = embedding_size  # Match base_model output size

class ModelWithEmbedding(nn.Module):
    def __init__(self, base_model, embedding_dim_input=512, embedding_dim_output=756):
        super().__init__()
        self.base_model = base_model
        self.embedding_layer = nn.Linear(embedding_dim_input, embedding_dim_output)
        self.relu = nn.ReLU()
        self.config = SimpleConfig(
            color_space='RGB',
            freeze=False,  # Set to True if you want to freeze base_model
            num_classes=None,  # Adjust based on your dataset
            embedding_size=embedding_dim_output  # Reflect the output embedding size
        )
        self._num_classes = self.config.num_classes  # Required by BaseModel

        # Ensure new layers are trainable
        self.embedding_layer.weight.requires_grad = True
        self.embedding_layer.bias.requires_grad = False

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x)
        #x = F.normalize(x, dim=1)
        return x

    def has_trainable_params(self):
        """Check if the model has any trainable parameters."""
        for p in self.parameters():
            if p.requires_grad:
                return True
        return False

    def num_classes(self):
        """Return the number of classes."""
        return self._num_classes

    def make_train_transform(self):
        """Delegate to base_model's train transform."""
        return self.base_model.make_train_transform()

    def make_test_transform(self):
        """Delegate to base_model's test transform."""
        return self.base_model.make_test_transform()