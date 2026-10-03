# Fisher LDA: a 45-minute numerical ML workshop

[English](fisher-lda.md) · [Русский](fisher-lda-ru.md)

Prepared by Jabrail Khalil. This is runnable teaching material, not a record of an event, mentoring session, attendance or external adoption.

Audience: Python developers familiar with matrix multiplication and basic differentiation. Prerequisites: Python 3.12, Git, a CPU, and the pinned environment in the [English article](../articles/generalized-eigh-lda.md#reproduce-the-lab) or [Russian article](../articles/generalized-eigh-lda-ru.md). No accelerator or paid cloud account is needed. Install and run the checks before the session; installation time is not included in the 45 minutes.

## Outcomes

By the end, participants should be able to explain what the two scatter matrices measure, distinguish ordinary from generalized Hermitian eigenproblems, test metric normalization, and compare an autodiff gradient with an independent finite-difference calculation.

## 0–8 minutes: identify the learning problem

Open [the example](../examples/differentiable_lda.py) and inspect `make_data` and `scatter_matrices`.

1. Which feature has high generating noise variance but no difference in the generating class means?
2. Why can PCA choose that feature while a label-aware projection de-emphasizes it?
3. For three classes, why can the between-class scatter have at most two independent directions?

Expected reasoning: PCA uses total variance rather than labels. The centered class means obey a weighted linear dependence. These observations explain the fixture; they do not establish held-out model quality.

## 8–18 minutes: solve the right equation

Write down $S_Bv=\lambda(S_W+\alpha I)v$ before running the code. Predict which orthogonality condition the resulting vectors satisfy.

Run from the repository root:

```sh
python examples/differentiable_lda.py
```

Find the reported eigenvalues, residual and normalization error. Trace how `check_dtype` calculates those quantities with an independent NumPy/SciPy reference.

Exercise: explain why symmetry of $B$ and $S_B$ does not make $B^{-1}S_B$ symmetric. Explain why checking only a plausible-looking projection can miss this error. Do not replace the Hermitian solver with an inverse-product shortcut in the shared example.

## 18–28 minutes: use JAX transformations without changing the question

Inspect the eager/JIT loop and the regularization sweep. Identify what is mapped by `vmap` and what stays fixed.

```sh
python examples/differentiable_lda.py --x64
```

Compare reported float32/float64 residuals and gradient errors. Explain why a change in floating-point precision does not justify demanding bitwise identical results or identical eigenvector signs. The lab checks invariants and tolerances instead.

## 28–40 minutes: make the gradient earn its keep

Inspect `score`, `scipy_score`, and the finite-difference loop.

1. Why parameterize ridge and feature scale in logarithms?
2. Does scaling the first feature change only $S_B$, or both scatter matrices?
3. Why is a NumPy class loop plus SciPy a stronger reference here than applying `jax.grad` twice to the same implementation?
4. What changes if the finite-difference step is too large or too small?

Expected reasoning: exponentiation preserves positivity; both matrices must be recomputed after scaling; the independent reference takes a different computational path; truncation and rounding errors compete in finite differences. The supplied $10^{-5}$ step is checked for this fixture, not asserted optimal for every problem.

## 40–45 minutes: explain the limits

Ask each participant to state one verified behavior and one unverified claim. Good answers distinguish numerical correctness on the synthetic CPU fixture from classification generalization, accelerator behavior, or eigenvector differentiation at repeated eigenvalues.

Optional extension in a personal copy: increase the third feature's noise and explain the projection change. Do not interpret a visually separated training projection as a measured test-set improvement.

## Organizer preparation and honest follow-up

Before a real session, run both commands on the presentation machine and retain their versioned output. Prepare an offline checkout with the environment already installed. Check the upstream PR/release status and retain the source pin if release availability is not verified.

For an event proposal: “A hands-on CPU lab showing how a merged JAX numerical API contribution becomes a tested ML computation: independent references, generalized eigenproblems, JIT/vmap and gradients.” Duration: 45 minutes after environment setup. Format: guided exercises and discussion; not a performance benchmark or a product pitch.

After a real session, record the public event/material link, date, actual delivery language, and concrete feedback or corrections. Record attendance only if reliable organizer evidence is available. Do not count this worksheet as an event or claim participants before one happens.
