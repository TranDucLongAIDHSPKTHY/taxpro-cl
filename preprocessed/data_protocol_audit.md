# DATA PROTOCOL AUDIT - TAXPRO-CL

**Protocol:** `taxprocl-p0-week3-v1`  
**Data split seed:** 42  
**Gate G1:** **PASS**

## Splits, hashes, and leakage

| Dataset | Split | Users | Items | Interactions | SHA-256 |
|---|---|---:|---:|---:|---|
| amazon-book | train | 52,642 | 88,416 | 2,159,968 | `644a62e3afb91bb6780efc1d50853361ca3cf5627993e2737580a83a0793cf06` |
| amazon-book | validation | 52,642 | 68,058 | 212,647 | `3b29ba64d32db26ba4376391b5cbf43a42311c0da08c9cba0cb7baddb0351899` |
| amazon-book | test | 52,639 | 82,629 | 603,378 | `6250754994dd953b91b48dccdd77a73f318e0634b4757c2abfcc9d33a54ab0f5` |
| yelp2018 | train | 31,668 | 37,381 | 1,126,670 | `0ab254dd0bec5914efc4a4a6d7f6bb318d234b5f9c64b0aa9449925d0ba81bf8` |
| yelp2018 | validation | 31,668 | 29,308 | 108,637 | `a0950d1880a909c41424b903945bcd1cf32b23181e076be92428a2a241b81c41` |
| yelp2018 | test | 31,668 | 36,073 | 324,147 | `9cd9b37926cd1447a06625ffa7914527573fb097c10cb0bbdb1d1b8ded3742fe` |

| Dataset | train/intersection/validation | train/intersection/test | validation/intersection/test | Test raw=verify |
|---|---:|---:|---:|---|
| amazon-book | 0 | 0 | 0 | PASS |
| yelp2018 | 0 | 0 | 0 | PASS |

The graph is built from training data. Validation excludes training positives; test excludes training and validation positives. Test data must not select taxonomy, thresholds, hyperparameters, epochs, or checkpoints.

## Group support

| Dataset | Group | Items | % catalog | Validation users/positives | Test users/positives |
|---|---|---:|---:|---:|---:|
| amazon-book | near_cold | 2,363 | 2.5797% | 2,349/2,643 | 13,238/21,296 |
| amazon-book | long_tail | 27,336 | 29.8431% | 16,210/27,365 | 36,609/112,142 |
| amazon-book | warm | 61,080 | 66.6820% | 50,378/185,282 | 52,123/442,510 |
| amazon-book | strict_cold | 3,183 | 3.4749% | 0/0 | 18,196/48,726 |
| yelp2018 | near_cold | 976 | 2.5652% | 1,074/1,119 | 5,217/6,528 |
| yelp2018 | long_tail | 10,659 | 28.0146% | 7,489/10,274 | 18,907/41,369 |
| yelp2018 | warm | 26,722 | 70.2323% | 30,702/98,363 | 31,661/275,460 |
| yelp2018 | strict_cold | 667 | 1.7530% | 0/0 | 5,333/7,318 |

Near-cold (1-5) is a subset of Long-tail (1-10); Warm (>10) is disjoint from Long-tail. Strict-cold (=0) is audit-only and is outside the three evaluation slices.

## Taxonomy coverage

| Dataset | Valid mapping IDs | Missing | Coverage |
|---|---:|---:|---:|
| amazon-book | 91,599 | 0 | 100.0000% |
| yelp2018 | 36,720 | 1,328 | 96.5097% |

## Environment

- Runtime audit: Python 3.10.0 (embeddable, standard library).
- Training readiness: BLOCKED in this shell: embeddable standard-library runtime can run protocol tests, but PyTorch/CUDA runtime is not available.

## Gate G1

**PASS**
