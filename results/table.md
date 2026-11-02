| arm | ckpt | affirm neutral | affirm sad | net sure flip | restate floor | gsm8k acc | warmth | mean tokens (syco) | invalid% (syco) | invalid% (ladder) |
|---|---|---|---|---|---|---|---|---|---|---|
| B | s0 | 0.48 [0.41, 0.54] | 0.43 [0.37, 0.49] | 0.39 [0.37, 0.42] | 0.06 [0.05, 0.08] | 0.80 [0.76, 0.83] | 1.94 [1.83, 2.05] | 145 [139, 151] | 13.6% [11.1, 16.2] | 0.0% [0.0, 0.0] |
| B | s40 | 0.46 [0.39, 0.52] | 0.37 [0.31, 0.43] | 0.46 [0.43, 0.49] | 0.07 [0.05, 0.09] | 0.84 [0.80, 0.88] | 1.86 [1.71, 1.99] | 137 [130, 143] | 15.8% [13.0, 18.8] | 0.0% [0.0, 0.0] |
| B | s80 | 0.46 [0.39, 0.52] | 0.40 [0.34, 0.46] | 0.45 [0.42, 0.48] | 0.08 [0.06, 0.10] | 0.84 [0.80, 0.87] | 1.87 [1.74, 1.98] | 131 [125, 137] | 13.5% [10.8, 16.4] | 0.0% [0.0, 0.0] |
| B | s120 | 0.48 [0.41, 0.54] | 0.40 [0.34, 0.46] | 0.48 [0.45, 0.51] | 0.08 [0.06, 0.10] | 0.86 [0.82, 0.89] | 1.85 [1.70, 1.98] | 128 [122, 134] | 13.2% [10.6, 16.1] | 0.0% [0.0, 0.1] |

Rates pooled over items × seeds; 95% bootstrap CIs (1,000 replicates, item ids resampled jointly across arms/ckpts/seeds/framings). Affirm rates exclude invalid/ambiguous rows (shown separately). Net sure flip = sure flip − restate flip.

| W−O gap | at s0 | at last ckpt | last − s0 | ratio last/s0 |
|---|---|---|---|---|
| affirm_sad | missing | | | |
| warmth | missing | | | |
| net_sure | missing | | | |
