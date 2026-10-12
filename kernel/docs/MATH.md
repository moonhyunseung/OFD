# Mathematical contract

For an orthogonal transform R and row-vector notation:

q_R = qR, K_R = KR, V_R = VR.

Then `(qR)(KR)^T = qK^T`. If `p = softmax(qK^T)`, then `(pVR)R^T = pV`.

Therefore four storage/execution variants are valid before quantization:

- P: plain K, plain V
- K: rotated K with rotated q, plain V
- V: plain K, rotated V with one inverse transform after aggregation
- KV: rotated K/V, rotated q, one inverse transform after aggregation

Quantization does not commute with rotation, so P/K/V/KV are numerically different after QDQ and must be measured rather than assumed equal.

## Stable split-KV merge

For split s, store `(m_s, l_s, o_s)` with

- `m_s = max z_i`
- `l_s = sum exp(z_i-m_s)`
- `o_s = sum exp(z_i-m_s) v_i`

Global merge uses `m=max_s m_s`, weights `a_s=exp(m_s-m)`, denominator `sum a_s l_s`, and numerator `sum a_s o_s`.

Empty splits are explicitly neutral: `m=-inf, l=0, o=0`. They must be masked before exponentiation to avoid `-inf - (-inf)` NaNs.
