import torch

class BatchAllTripletLoss(torch.nn.Module):
    def __init__(self, margin:float, device:torch.device):
        """Compute the Batch All triplet loss method.        
        """
        super(BatchAllTripletLoss, self).__init__()        
        self.margin = margin
        self.device = device

    def _pairwise_distances(self, embeddings:torch.Tensor)->torch.Tensor:  
        """Compute the 2D matrix of distances between all the embeddings.

        -------------------
        Parameters:
            embeddings: torch.Tensor of shape (batch_size, embed_dim)
        -------------------
        Returns:
            imgs_crop: torch.Tensor of shape (batch_size, batch_size)
        """
        embeddings = embeddings.to(torch.float)
        distances = torch.cdist(embeddings, embeddings, p=2)
        return distances

    def _get_triplet_mask(self, labels:torch.Tensor)->torch.Tensor:
        """Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]

        -------------------
        Parameters:
            labels: torch.int32 of shape (batch_size,)
        -------------------
        Returns:
            mask: torch.Tensor of shape (batch_size, batch_size, batch_size)
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.shape[0]).to(torch.bool).to(self.device)
        indices_not_equal = torch.logical_not(indices_equal)
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

        # Combine the two masks
        mask = torch.logical_and(distinct_indices, valid_labels)
        return mask

    def forward(self, labels:torch.Tensor, embeddings:torch.Tensor):
        """Forward method of the class. Build the triplet loss over a batch of 
        embeddings. All the valid triplets are generate and average the loss 
        over the positive ones.

        -------------------
        Parameters:
            labels: torch.Tensor of shape (batch_size,)
                Labels of the batch
            embeddings: torch.Tensor of shape (batch_size, embed_dim)
                Embeddings of the batch
        -------------------
        Returns:
            triplet_loss: torch.float, scalar 
                Contains the triplet loss of the batch
            fraction_positive_triplets: flot, scalar
                Contains the ratio of positive triplets in the batch
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask(labels).to(torch.float)
        triplet_loss = torch.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))
        
        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = torch.gt(triplet_loss, 1e-16).to(torch.float)
        num_positive_triplets = torch.sum(valid_triplets)
        num_valid_triplets = torch.sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets
