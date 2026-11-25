# CPU reference instruments

These interfaces define deterministic data, scoring and packed reconstruction for design review. They do not include an I1 training loop, GPU kernels, a model evaluator or measurements of learned capacity. The focused tests use NumPy on CPU and temporary local Git repositories.

```text
python -B -m pytest -q -p no:cacheprovider tests/test_cpu_instruments.py tests/test_research_protocol.py
```

## Worlds and information

`world.py` derives a per-world key with HMAC-SHA256 using a private 32-byte root key and the message `domovoi/world/v1\0 || key\0 || seed:u64 || identities:u32 || attributes:u16`. Integers are unsigned big-endian. The `random-fact` draw uses the world key and the framed identity and attribute (`u16` each). The first two digest bytes are the label. A separate domain generates untrained labels. A published world ID commits to the world key via SHA-256. The test key `00..1f`, seed `1234`, 20 identities and four attributes yield `(92, 182)` for identity 3, attribute 2. Generate and seal a fresh root for research. The test key is public.

The training record has seven tokens. A held-out formatting probe changes symbol and order while preserving the fact. The untrained namespace never shares the supervised label domain. The structured split holds back every fifth identity and asks the model to apply an attribute-specific affine rule to the ID bytes. The two-hop probe composes a separately generated link with a known attribute fact. Neither derived probe contributes independent 16-bit facts to K. Keep `World`, held-out labels and key files on the data/evaluator side. Pass only `Query` or the intended supervised record across the boundary. Teacher-force one true byte only when scoring the conditional second-byte distribution. Free decoding never sees it. These types prevent accidental serialization of the generator. The future harness must enforce process and file separation.

`information.py` computes `K = 16N + Σ log₂ q(y₁|x) + Σ log₂ q(y₂|x,y₁)` with a normalized 256-way distribution per byte. It retains negative values. `K*` fits one temperature on a disjoint calibration world or declared disjoint identities in the same trained world. The first maximum wins on this fixed grid.

```text
0.25 0.35 0.5 0.65 0.8 0.9 1.0 1.1 1.25 1.5 1.75 2.0 2.5 3.0 4.0 8.0
```

The result includes raw and calibrated K per actual whole-representation bit, raw free-decoding invalid rate and exact match, and 10 equal-width bin byte ECE at both temperatures. Paired uncertainty summarizes calibrated-density differences by independent seed/world pair with sample standard deviation and a descriptive paired bootstrap. Three pairs provide an instrument check with limited resolution.

For an ideal independent uniform fact population and a fixed predictor, `E[2^K] = 1`. A union bound over at most `2^S` distinct S-bit representations gives `Pr(K > S + t) ≤ 2⁻ᵗ`. The scorer uses `t=16` per checked endpoint and charges `log₂(16)=4` extra bits when the temperature was selected in the scored world. It flags possible leakage or undercounted state. Include every world-dependent deployed bit in S and freeze the query set, decoder, model selection, number of worlds and hypothesis family before interpreting this check. A family-wide guarantee needs a multiplicity budget. The generator relies on HMAC behaving as a PRF and is a finite-seed simulation of the ideal argument. A clean alarm does not rule out leakage. Uniform logits give K=0 and an exact oracle approaches 16N. The tests cover permutation, untrained labels, deliberate answer leakage, incorrect entropy and overconfident wrong logits.

## Packed matrices

The format starts with a 16-byte little-endian `DMC1` header with format ID and two dimensions. The header and each nonempty payload segment are zero-padded to 64-byte boundaries. Quantized formats carry a positive little-endian FP16 scale per 64 flattened weights. Eight-coordinate blocks pad their tail to eight. The codec accounts for every serialized byte.

| Format | Payload before scales, tables and alignment |
|---|---|
| BF16 | 16 bits per value, round to nearest even from FP32 |
| INT8 | 8 bits per value, symmetric signed −127 through 127 |
| INT4 | 4 bits per value, low nibble first, code `weight+7` for −7 through 7 |
| Scalar 2-bit | 2 bits per value, low bits first, levels −3, −1, 1, 3 |
| Ternary | Five base-3 digits per byte, least significant first, digit `weight+1`, 1.6 bits/value before tails |
| Rotated scalar 2-bit | Scalar 2-bit in an order-eight normalized Sylvester Hadamard basis |
| E8 core v1 | 16-bit little-endian index per eight values |
| Fixed or learned residual VQ | Two 8-bit indices per eight values and two 256×8 FP16 tables |

The finite E8 core uses `D₈ ∪ (D₈ + ½·1)`, with `D₈` the even-sum integer lattice. Enumerate doubled coordinates with squared norm at most 48 and sum divisible by four. Put zero at index 0. For each antipodal pair choose the representative whose first nonzero coordinate is positive. Sort by squared norm then lexicographically and emit the first 32,767 representatives, each followed by its negative. Index 65,535 is reserved. This is a specified CPU reference format distinct from QuIP# and E8-QAT bitstreams. Bill at least the `65,535×8` bytes if the doubled-coordinate lookup is retained, and bill any wider or duplicate resident table as well.

In the fixed two-stage VQ, each row of a table is indexed by a sign vector `s∈{−1,+1}⁸`, least-significant bit first. With the same orthogonal basis `Q` in both stages, the sum decodes as `Q(s₁+½s₂)=Q({−1.5,−0.5,0.5,1.5}⁸)`. Greedy nearest stage selection yields the scalar decision boundaries at `−1, 0, +1` in each rotated coordinate, up to deterministic boundary ties. That arm is a rotated four-level scalar code. With basis `I` in stage one and normalized Hadamard `H` in stage two, the hull of all decoded codepoints is the Minkowski sum of two orthogonally oriented cubes. Its 16 distinct generator and edge directions differ from the eight directions of a rotated scalar product hull. FP16 table storage multiplies the Hadamard directions by a common factor and preserves this distinction. The tests also exhibit distinct greedy reconstructions. Hull geometry describes the decodable set and does not depend on every index pair being reached by greedy assignment. The learned variant accepts supplied tables with the same layout and does not fit them.

`bytes.py` receipts separate packed values, scales, tables, metadata and alignment from embeddings, attention, norms, lookups and duplicated allocations. Charge tied allocations once and independently retained shadows in full. Serialized/persistent representation bytes are separate from measured device peak, which includes activations, runtime and workspace. Tests compare decoded logits against independently accumulated decoded dot products. Approximation error to the original weights is a different quantity.

## Fixed-tensor control for the capacity study

Freeze the MLP at a world-independent preregistered initialization while training embeddings, attention and norms with the matched recipe, data order and seed. Keep the forward path, count all fixed and trained resident bytes, report a separately declared fixed-tensor control and inspect its learning curves. Report whole-model K and `(K_full−K_fixed)/coded_matrix_bytes` side by side. Subtraction is a descriptive ablation contrast because the components interact during optimization. It does not prove the information was localized in one allocation.
