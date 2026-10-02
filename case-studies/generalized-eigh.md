# Generalized eigenproblems: numerical identities and transformations

I added type-1 generalized Hermitian eigenproblems to JAX in [#41162](https://github.com/jax-ml/jax/pull/41162), merged on 1 October 2026, and types 2 and 3 in [#41197](https://github.com/jax-ml/jax/pull/41197), merged on 2 October. Jake VanderPlas (`jakevdp`) reviewed both changes. They address [#5461](https://github.com/jax-ml/jax/issues/5461).

This is a CPU lab for studying those changes. It is prepared teaching material; I do not claim a delivered workshop or downstream adoption.

## Three related problems

Let A be Hermitian and B be Hermitian positive definite. With B = L Lᴴ, the three problem types are:

| `type` | Defining equation | Eigenvector normalization |
| --- | --- | --- |
| 1 | A v = λ B v | Vᴴ B V = I |
| 2 | A B v = λ v | Vᴴ B V = I |
| 3 | B A v = λ v | Vᴴ B⁻¹ V = I |

For type 1, reduce to the Hermitian matrix C = L⁻¹ A L⁻ᴴ and solve C u = λ u. Recover v = L⁻ᴴ u. The implementation uses triangular solves rather than explicitly forming an inverse.

Types 2 and 3 share C = Lᴴ A L, so they have the same eigenvalues for the same A and B. Their eigenvectors differ: type 2 uses v = L⁻ᴴ u, while type 3 uses v = L u. Reusing the type-2 back-transform for type 3 would preserve the eigenvalues while violating the defining equation and normalization. An eigenvalues-only comparison would miss that bug.

## What the lab checks

[The example](../examples/generalized_eigh.py) generates deterministic real and complex Hermitian matrices and an SPD B. For every problem type, it checks:

- Eigenvalues against SciPy, in eager and JIT execution.
- The defining equation and the correct normalization, without relying on eigenvector signs or complex phases.
- The eigenvalues-only API and a JIT-compiled vmap over two input matrices with a shared B.
- A spectral objective, Σ log(1 + λ²), differentiated with respect to parameters in both A and B. An independent SciPy central difference checks both derivatives.
- Non-positive-definite B: the current API returns NaN eigenvalues.

The derivative fixture has separated eigenvalues. These checks do not establish differentiability of eigenvectors at repeated eigenvalues, robustness for all condition numbers, or accelerator behavior.

## Run it

As of 2 October 2026, the latest published JAX release is 0.11.2, which predates these changes. Use the tested source revision in a separate environment; the older before/after labs use a different source checkpoint.

From the repository root, create and activate a fresh Python 3.12 environment, then run:

```sh
git clone https://github.com/jax-ml/jax.git .lab/generalized-jax
git -C .lab/generalized-jax checkout 15998e8040d07995ac5dcc4643333e1b88e8c047
python -m pip install -r requirements-generalized-eigh.txt .lab/generalized-jax
python examples/generalized_eigh.py
python examples/generalized_eigh.py --x64
```

The program exits on a failed assertion and otherwise prints the runtime versions, devices, residuals and gradient comparisons as JSON. The default run covers six dtype/problem-type combinations; `--x64` covers twelve. I ran both on Windows CPU with Python 3.12.10 and jaxlib 0.11.2. Measured outputs are in [validation/generalized-eigh-x32.json](../validation/generalized-eigh-x32.json) and [validation/generalized-eigh-x64.json](../validation/generalized-eigh-x64.json).

The same six/twelve-case runs also passed in [Linux CPU CI](https://github.com/jabrailkhalil/jax-reliability-notes/actions/runs/37032767112), with Python 3.12.14. That run also passed the existing keyword-forwarding and uneven-group before/after examples.

## A 45-minute exercise

1. Derive the type-1 Cholesky reduction and explain why B must be positive definite (10 minutes).
2. Run the three problem types and compare their residuals and normalization (10 minutes).
3. Replace the type-3 back-transform on paper with the type-2 transform. Identify which assertions would detect the mistake (5 minutes).
4. Examine the two parameter derivatives. Change the finite-difference step and compare float32 with float64 (10 minutes).
5. Discuss repeated eigenvalues, conditioning, and the difference between CPU evidence and accelerator validation (10 minutes).

The same lab can support an HSE class, a workshop segment, or an article on testing numerical APIs. Audience and adoption evidence should be recorded after those activities happen.
