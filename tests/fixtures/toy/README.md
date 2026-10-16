# Toy oracle set (B3, T3)

Hand-checked data for the oracle test in `tests/eval/test_real_data.py`.

## Files

| File | Content |
|---|---|
| `gold/Bed017.jsonl` | positions 290–299 of val meeting Bed017, copied unchanged from `data/v4_meetings` |
| `gold/Bmr003.jsonl` | positions 1759–1768 of val meeting Bmr003, copied unchanged |
| `gold/Bmr024.jsonl` | positions 1080–1089 of val meeting Bmr024, copied unchanged |
| `predictions.jsonl` | the same 30 segments in the B1 prediction format, with 5 labels changed by hand |

## Gold count per class

| cs | co | aa | bk | ar | cc | other | sum |
|---|---|---|---|---|---|---|---|
| 8 | 3 | 2 | 7 | 1 | 2 | 7 | 30 |

## Hand edits (gold → predicted)

| seg_id | gold | predicted | case covered |
|---|---|---|---|
| `Bed017-c5_0914398_0918558` | cc | cs | wrong target |
| `Bed017-c1_0936730_0936950` | bk | aa | wrong target |
| `Bmr003-c0_3277070_3283820` | other | aa | gold `other` predicted as a target |
| `Bmr024-cB_1875090_1875480` | ar | bk | `ar` never predicted (zero division) |
| `Bmr024-cB_1888860_1893420` | cs | co | cs / co confusion |

## Expected values (computed by hand)

F1 = 2PR / (P + R)

| Class | Gold | Pred | Correct | P | R | F1 |
|---|---|---|---|---|---|---|
| cs | 8 | 8 | 7 | 7/8 | 7/8 | 7/8 |
| co | 3 | 4 | 3 | 3/4 | 1 | 6/7 |
| aa | 2 | 4 | 2 | 1/2 | 1 | 2/3 |
| bk | 7 | 7 | 6 | 6/7 | 6/7 | 6/7 |
| ar | 1 | 0 | 0 | 0 | 0 | 0 |
| cc | 2 | 1 | 1 | 1 | 1/2 | 2/3 |
| other | 7 | 6 | 6 | 1 | 6/7 | 12/13 |
| **Total** | **30** | **30** | **25** | | | |

- Accuracy = 25/30
- **Macro-F1 (6)** (primary, excludes `other`) = (7/8 + 6/7 + 2/3 + 6/7 + 0 + 2/3) / 6 = 8567/13104 ≈ 0.6538
- Macro-F1 (7) = (7/8 + 6/7 + 2/3 + 6/7 + 0 + 2/3 + 12/13) / 7 = 10583/15288 ≈ 0.6922
