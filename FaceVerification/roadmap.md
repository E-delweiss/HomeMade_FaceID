# Introduction
We are talking about verification that is, verified that a person is the expected person *-is it this person ?-* One refers to this as a **1:1 problem** i.e. two faces are shown to the model and it should be able to assert if they belong to the same person or not. This type of problem is the same for Apple iPhone FaceID or airport passport verification with a "calibration phase" (when you use the phone for the first time to specialized the algorithm with your face, or when you are invited to present your passport to the scan so it can verify if you are indeed the person on the passport photo).

At the contrary, a face recognition problem is a **1:K problem** : the model tries to find if it recognizes the person (if this person if a member of the database used to train the model). It involves a representative dataset for each member and a certain flexibility since we expect that the model should handle a new person whithout retraining it from scratch. One can find this type of application in some company entrence for instance.

Here, we are trying to unlock a computer with the administrator face. Since no one else will be authorized to unlock the session, it's a **face verification 1:1 problem**.

# What should the model do ?
The model should authorized one specific person and rejects all different. To achive this, we'll use the technique of **one shot learning with Siamese Networks** which is a well know technique for face verification.

One Shot Learning is the idea in which classification or categorization tasks can be achieved with one or few examples to classify many new examples. Back to our problem, an image of the administrator will be given to the model (it's the *anchor*) and the model should be capable to classify any image regarding this anchor (this is the same person or it is not).

Technically, the model will encode the image as an embedding vector of d-dimension (e.g. 128). That is, any embedding image will be place in a n-dimensional Euclidean space and will belong to a specific location such that (luckily) similar identities should be close to each others, and different identities will be found far apart.

In the [MNIST project](https://github.com/E-delweiss/HomeMade_FaceID/tree/main/ImageVerification_MNIST), I show how to do an image classification with 10 classes (where all different digits embedding are far from each other) and with only 2 classes (where all non-7 digit embeddings are far from the 7 embedding digits).

