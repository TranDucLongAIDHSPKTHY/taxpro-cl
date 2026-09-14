# TaxPro-CL

TaxPro-CL combines a LightGCN recommender objective with taxonomy-guided item
views and item-level contrastive learning. Taxonomy assignments are locked from
training evidence. Items with invalid taxonomy remain in BPR training and the
full evaluation catalog but are excluded from the taxonomy contrastive term.

This implementation covers leaf prototypes, EMA stop-gradient updates, controlled
item perturbation, and InfoNCE. Sibling loss, gating, and multi-level prototype
extensions are outside its scope.
