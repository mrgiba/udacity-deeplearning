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
            nn.Conv2d(3, 64, 3, padding=1),       # output size: (224 - 3 + 2*1) + 1 = 224            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # output size: (224/2) = 112            

            nn.Conv2d(64, 128, 3, padding=1),     # output size: (112 - 3 + 2*1) + 1 = 112           
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # output size: (112/2) = 56            

            nn.Conv2d(128, 192, 3, padding=1),    # output size: (56 - 3 + 2*1) + 1 = 56
            nn.BatchNorm2d(192),            
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # output size: (56/2) = 28
            
            nn.Conv2d(192, 256, 3, padding=1),   # output size: (28 - 3 + 2 *1) + 1 = 28
            nn.BatchNorm2d(256),   
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # output size: 14x14

            nn.Conv2d(256, 384, 3, padding=1),   # output size: (14 - 3 + 2 *1) + 1 = 14
            nn.BatchNorm2d(384),   
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # output size: 7x7

            # Flatten feature maps
            nn.Flatten(),            
            nn.Linear(7 * 7 * 384, 4096),
            # Batch normalization not called to avoid 'Expected more than 1 value per channel when training, got input size torch.Size([1, 4096])' errors on small batch sizes
            # nn.BatchNorm1d(4096), 
            nn.Dropout(p=dropout),            
            nn.ReLU(),
                                    
            nn.Linear(4096, 4096),
            # nn.BatchNorm1d(4096),
            nn.Dropout(p=dropout),            
            nn.ReLU(),
                        
            nn.Linear(4096, num_classes)            
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
