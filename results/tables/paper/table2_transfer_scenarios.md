**Table 2. Transfer scenarios**

| Code | Scenario | Training | Test | Feature_set | Bag_set | N_bags | n_individuals | Note |
|---|---|---|---|---|---|---|---|---|
| S0 | random seconds (leaky reference) | Smit collar, random 70 % of seconds | random 30 % of seconds | D17 | Smit 30-min bags | 68 | 12 | leaky by definition; not an evaluation of transfer |
| S1 | new individual | Smit collar, 11 cats | held-out cat | D17 | Smit 30-min bags (>=50 % coverage) | 68 | 12 | also natural 2-min (1083 bags) and APP 120-s (600 bags) |
| S1-L2 | new individual, 7-class space | Smit collar, 11 cats | held-out cat | 32 features | Smit 30-min bags | 68 | 12 |  |
| S2 | harness → collar | Smit harness, 11 cats | held-out cat (collar) | D17 | Smit 30-min bags | 68 | 12 |  |
| S3 | cross-study (Smit → Dunford) [exploratory] | Smit collar, 12 cats | Dunford cats | D17 | Dunford cat bags (N>=60) | 8 | 8 |  |
| S4-Smit-open | dog → cat, open set | 45 dogs, collar (2 classes) | Smit cats (3 classes; Care unseen) | D17 | Smit 30-min bags | 68 | 12 |  |
| S4-Smit-closed | dog → cat, closed set | 45 dogs, collar (2 classes) | Smit cats, Care seconds removed | D17 | Smit 30-min bags | 68 | 12 |  |
| S4-Dunford-open / closed | dog → cat [exploratory] | 45 dogs, collar | Dunford cats | D17 | Dunford cat bags (N>=60) | 8 / 7 | 8 / 7 |  |
| K-L1, K-L2 | within-dog (new dog) | dogs, 5-fold GroupKFold | held-out dogs | 32 features (D17 as sensitivity) | dog × recording (61 bags) and natural 2-min (130 bags) | 61 / 130 | 44 / 35 |  |
| R-D | Dunford leave-one-cat-out [exploratory] | Dunford cats | held-out cat | D17 | Dunford cat bags (N>=60) | 8 | 8 |  |
| S1-closed | new individual, Care removed | Smit collar, 11 cats | held-out cat | D17 | Smit 30-min bags | 68 | 12 |  |

Bag sets follow the primary bag-set rule of the project; Dunford rows are exploratory.
