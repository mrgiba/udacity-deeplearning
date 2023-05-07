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

        # Alexnet - https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py        
         # VGG-16 - https://gist.github.com/KushajveerSingh/7773052dfb6d8adedc53d0544dedaf60    
         #  https://towardsdatascience.com/what-your-validation-loss-is-lower-than-your-training-loss-this-is-why-5e92e0b1747e
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),       # output size: (224 - 3 + 2*1) + 1 = 224            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # output size: (224/2) = 112
            
            # nn.Dropout2d(p=dropout),

            nn.Conv2d(64, 128, 3, padding=1),     # output size: (112 - 3 + 2*1) + 1 = 112           
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),     # output size: (112 - 3 + 2*1) + 1 = 112           
            nn.BatchNorm2d(128),
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),                            # output size: (112/2) = 56

            
            # nn.Dropout2d(p=dropout),

            nn.Conv2d(128, 192, 3, padding=1),    # output size: (56 - 3 + 2*1) + 1 = 56
            nn.BatchNorm2d(192),            
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1),     # output size: (56 - 3 + 2*1) + 1 = 56           
            nn.BatchNorm2d(192),            
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # output size: (56/2) = 28
            
            
            # nn.Dropout2d(p=dropout),
            
            nn.Conv2d(192, 256, 3, padding=1),   # output size: (28 - 3 + 2 *1) + 1 = 28
            nn.BatchNorm2d(256),   
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),   # output size: (28 - 3 + 2 *1) + 1 = 28         
            nn.BatchNorm2d(256),   
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),                            # output size: 14x14
            # nn.Dropout2d(dropout),

            nn.Conv2d(256, 384, 3, padding=1),   # output size: (14 - 3 + 2 *1) + 1 = 14
            nn.BatchNorm2d(384),   
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),   # output size: (14 - 3 + 2 *1) + 1 = 14
            nn.BatchNorm2d(384),   
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),                            # output size: 7x7

            # Flatten feature maps
            nn.Flatten(),            
            nn.Linear(7 * 7 * 384, 4096),             
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(4096), 
            nn.ReLU(),
                                    
            nn.Linear(4096, 4096),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(4096),
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
