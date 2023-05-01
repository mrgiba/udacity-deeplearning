import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        #  Input size: 224x224        
        #  Feature map channel output size = (input_size - kernel_size + 2*padding)  + 1

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),       # output size: (224 - 3 + 2*1) + 1 = 224
            nn.MaxPool2d(2, 2),                            # output size: (224/2) = 112
            nn.ReLU(),
            # nn.Dropout2d(p=dropout),

            nn.Conv2d(16, 32, 3, padding=1),     # output size: (112 - 3 + 2*1) + 1 = 112
            nn.MaxPool2d(2, 2),                            # output size: (112/2) = 56
            nn.ReLU(),
            # nn.Dropout2d(p=dropout),

            nn.Conv2d(32, 64, 3, padding=1),    # output size: (56 - 3 + 2*1) + 1 = 56
            nn.MaxPool2d(2, 2),                           # output size: (56/2) = 28
            nn.ReLU(),
            # nn.Dropout2d(p=dropout),
            
            nn.Conv2d(64, 128, 3, padding=1),   # output size: (28 - 3 + 2 *1) + 1 = 28
            # nn.MaxPool2d(2, 2),                            # output size: 14x14
            nn.ReLU(),
            # nn.Dropout2d(dropout),

            nn.Conv2d(128, 256, 3, padding=1),   # output size: (28 - 3 + 2 *1) + 1 = 28
            # nn.MaxPool2d(2, 2),                            # output size: 14x14
            nn.ReLU(),

            # Flatten feature maps
            nn.Flatten(),
           
            nn.Linear(28 * 28 * 256, 128), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)  


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
