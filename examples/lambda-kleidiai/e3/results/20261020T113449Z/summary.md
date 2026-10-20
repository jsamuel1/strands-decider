| arm | strategy | status | cold download p50 s (n) | vs ctl p50 s [95% CI] | MiB/s p50 | GETs | peak RSS after dl MB | /tmp MB | hash | answers identical | compute moved | status within largest CPU part |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ldr | `loader` | no-change | 25.03 (50) | +0.08 [-0.00, +0.16] | 79.1 | [31] | 599.4 | 0.0 | ok | 96/96 | none | no-change (0xd40, n=41) |
| aa | `threads:64:32` | no-change | 25.03 (50) | +0.07 [-0.00, +0.16] | 79.1 | [31] | 597.5 | 0.0 | ok | 96/96 | none | no-change (0xd40, n=40) |
| t16x64 | `threads:16:64` | regressed | 25.31 (50) | +0.56 [+0.19, +1.73] | 77.0 | [124] | 861.0 | 0.0 | ok | 98/98 | none | regressed (0xd40, n=48) |
| t8x128 | `threads:8:128` | regressed | 36.31 (50) | +11.40 [+10.15, +11.86] | 54.4 | [248] | 849.9 | 0.0 | ok | 98/98 | none | regressed (0xd40, n=46) |
| t32x64 | `threads:32:64` | regressed | 25.13 (50) | +0.17 [+0.08, +0.33] | 78.8 | [62] | 962.1 | 0.0 | ok | 98/98 | none | regressed (0xd40, n=40) |
| t64x64 | `threads:64:64` | no-change | 24.99 (50) | +0.03 [-0.07, +0.11] | 79.3 | [31] | 597.4 | 0.0 | ok | 159/159 | none | no-change (0xd40, n=44) |
| crt8 | `crt:8:25` | regressed | 27.21 (50) | +2.27 [+1.48, +4.70] | 72.7 | [248] | 1475.0 | 0.0 | ok | 99/99 | cold.model_load_mmap -0.245s | regressed (0xd40, n=44) |
| crt16 | `crt:16:25` | regressed | 25.70 (50) | +0.74 [+0.54, +1.48] | 77.0 | [124] | 1875.2 | 0.0 | ok | 96/96 | cold.model_load_mmap -0.336s | regressed (0xd40, n=39) |
| crtf16 | `crtfile:16:25` | regressed | 25.14 (50) | +0.18 [+0.07, +0.69] | 78.8 | [124] | 1079.1 | 0.0 | ok | 97/97 | cold.model_load_mmap -0.194s | regressed (0xd40, n=39) |
