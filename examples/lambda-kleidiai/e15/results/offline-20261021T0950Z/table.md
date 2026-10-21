| Category | Share of warm graph wall time (per round) | ms per forward (per round) | Nodes per forward |
|---|---|---|---|
| KleidiAI matmul (Q8_0 weights, i8mm) | 60.47% / 60.46% | 141.19 / 141.57 | 186 |
| GATED_DELTA_NET | 9.62% / 9.59% | 22.46 / 22.46 | 18 |
| attention KQV (F32 MUL_MAT, ggml_vec_dot_f32) | 5.40% / 5.39% | 12.60 / 12.62 | 6 |
| attention KQ (F32 MUL_MAT, ggml_vec_dot_f32) | 5.21% / 5.20% | 12.16 / 12.17 | 6 |
| CONCAT (E14) | 4.93% / 5.01% | 11.52 / 11.73 | 18 |
| Hadamard rotation (MUL_MAT -> ggml_compute_forward_fwht) | 4.39% / 4.39% | 10.26 / 10.28 | 96 |
| SSM_CONV | 2.65% / 2.63% | 6.18 / 6.17 | 18 |
| RMS_NORM (+ fused MUL) | 1.97% / 1.96% | 4.60 / 4.59 | 115 |
| SWIGLU | 1.12% / 1.13% | 2.63 / 2.64 | 24 |
| SILU | 1.05% / 1.04% | 2.46 / 2.45 | 36 |
| other: MUL | 0.53% / 0.54% | 1.23 / 1.25 | 138 |
| other: SOFT_MAX | 0.46% / 0.46% | 1.07 / 1.07 | 6 |
| other: GET_ROWS | 0.42% / 0.42% | 0.98 / 0.99 | 39 |
| other: CPY | 0.40% / 0.40% | 0.92 / 0.93 | 72 |
| other: SCALE | 0.36% / 0.36% | 0.84 / 0.85 | 72 |
| other: ADD | 0.33% / 0.32% | 0.76 / 0.75 | 66 |
| other: SIGMOID | 0.29% / 0.29% | 0.67 / 0.68 | 24 |
| other: SET_ROWS | 0.13% / 0.13% | 0.30 / 0.31 | 12 |
| other: ROPE | 0.12% / 0.12% | 0.27 / 0.27 | 12 |
| other: CONT | 0.08% / 0.08% | 0.20 / 0.19 | 12 |
| other: SOFTPLUS | 0.03% / 0.03% | 0.06 / 0.06 | 18 |
