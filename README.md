[repo in process]
# Overview
Working on a FaceID-like (low fidelity) app that unlock your computer.
This project uses deep learning through three main steps :
- Warming up with MNIST
- Detecting faces
- Verifying face
- Unlocking computer via shell script.

The first part has been done on MNIST dataset to understand the concept of Siamese Networks: same  networks are used to turn images into embeddings (128 dimension encoding vectors). Each embedding vector represents (*encodes*) an image. The more 2 images are different, the larger the difference between their embedding vectors. Training has been performed with **Online Triplet Mining**.

This [first project](https://github.com/E-delweiss/HomeMade_FaceID/tree/main/imageVerification_MNIST) on MNIST gives visualizations in 2D space thanks to the TSNE algorithm. There are two results: 
* the first shows how the model takes apart **each handwritting digits from each other**, 
* the second shows how the model takes apart **all the digit (label is 0) regarding a chosen one** (here the 7).

<p align="center">
  <img src="imageVerification_MNIST/results/BatchAll MNIST distribution (test set).png?raw=true" alt="balanced_set" width="300"/>
  <img src="imageVerification_MNIST/results/BatchAll MNIST distribution unbalanced (test set).png?raw=true" alt="unbalanced_set" width="300"/>
</p>
