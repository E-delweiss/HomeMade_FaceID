# Introduction
This GitHub repo is about face verification. To tackle this subject, I chose to start with the MNIST dataset. The goal here is to understand how to use the *Siamese Network* method. At the end, each digit of the MNIST dataset is expected to be encoding into a 128d embedding vector where each digit embedding vector is grouped in a 128d space with its fellows.

The concepts won't be explain in detail here but in the [main project '#TODO']().

# Dataset
The MNIST dataset from `torchvision` is used. Each image is turned into `torch.Tensor` and normalize with `(mean, std) = (0.1307, 0.3081)`.

To treat the *unbalanced* case, the **7 digit** has been chosen so, when the dataloader calls the dataset class, an image of a 7 has the label 7 but the label of any other image will be 0.
