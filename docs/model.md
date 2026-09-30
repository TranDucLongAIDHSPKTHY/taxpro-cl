# TaxPro-CL

TaxPro-CL (`models/TaxPro-CL.py`) keeps SimGCL's LightGCN backbone and two-view
training: each batch runs one clean pass (BPR plus the ego-embedding L2
regularizer) and two independent perturbed passes (InfoNCE), with the mean of
the propagated layers 1..L (layer 0 excluded) as the embedding. It changes the
item-side perturbation:

- **Direction (Eq. 1):** inside every propagation layer, an item moves toward
  the exponential-moving-average prototype of its taxonomy leaf,
  `(p_l - e_i) / (||p_l - e_i|| + delta)`, instead of SimGCL's sign-aligned
  random direction. Prototypes are updated from the clean embeddings of the
  batch's positive items (no gradient) before the two perturbed passes.
- **Magnitude (Eq. 2):** `epsilon_i ~ U(0, epsilon_max / L * s(deg_i))`, with a
  step function `s` of the train-time degree (`gamma_cold` for degree 1-5,
  1 for 6-10, `gamma_warm` above 10).
- **Users:** SimGCL's sign-aligned random perturbation with `epsilon_user`, and
  a user-side InfoNCE term with its own temperature (`use_user_ssl`).
- **Warm-start:** the first `warm_start_epochs` epochs train BPR plus the L2
  regularizer only (no perturbed pass, no InfoNCE); the
  prototypes are initialized once, immediately before the first joint epoch.

Items without a valid leaf, and items with no training interaction, get a zero
direction and are left out of the item-side InfoNCE term; they remain in BPR
training and in the full evaluation catalog. Online Resource 1, Figure S1 shows
one training step and the per-layer perturbation next to SimGCL's.

Sensitivity-check options (`prototype_mode=parent|mixture`, `isotropic_blend`,
`same_leaf_weight`, `prototype_weighting=leaf_uniform`,
`prototype_leave_one_out`, `asymmetric_view_direction`,
`augmentation_direction=random`, `use_adaptive_epsilon=false`) default to the
main configuration. Sibling losses, gating, and multi-level prototype memories
are outside the scope of this implementation.
