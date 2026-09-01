| SAE Ratio | Algorithm | Mean Acc. (%) | Worst-Group (%) | \|ΔTPR\| (%) | \|ΔFPR\| (%) | Seeds |
|---|---|---|---|---|---|---|
| 99.0% | ERM (rand.) | 99.95 ± 0.11 | 99.54 ± 0.99 | 0.00 ± 0.00 | 0.12 ± 0.25 | 10 |
|  | ERM (GB) | 99.92 ± 0.08 | 99.65 ± 0.37 | 0.07 ± 0.11 | 0.06 ± 0.15 | 10 |
|  | G. DRO | 99.99 ± 0.01 | 99.92 ± 0.17 | 0.00 ± 0.00 | 0.02 ± 0.03 | 10 |
| 99.5% | ERM (rand.) | 97.72 ± 2.94 | 84.58 ± 19.71 | 0.02 ± 0.05 | 5.24 ± 6.80 | 10 |
|  | ERM (GB) | 98.70 ± 2.41 | 90.79 ± 16.61 | 0.00 ± 0.00 | 3.00 ± 5.56 | 10 |
|  | G. DRO | 98.73 ± 2.56 | 91.50 ± 17.23 | 0.00 ± 0.00 | 2.93 ± 5.92 | 10 |

**Table 8 (revised): Mitigating the dialect penalty under data imbalance.** Mean/worst-group accuracy plus dialect-penalty metrics (|ΔTPR|, |ΔFPR|; SAE-relative). The penalty (FPR-side) only emerges under extreme SAE imbalance (≥99%); GroupDRO degrades more gracefully than ERM there. ΔTPR≈0 throughout (toxic prompts are topically salient). Mean ± std over 10 seeds.
