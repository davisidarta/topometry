# Mathematical Details

TopoMetry is built on a family of geometric operators rooted in differential geometry and spectral theory. This page gives an accessible overview of the core ideas — from why high-dimensional biology lives on a curved surface, to how TopoMetry learns that surface from data and why this is fundamentally better than approaches based on global variance (PCA, scVI, etc.).

No prior knowledge of differential geometry is required. Where equations appear, we explain what they mean in plain language right next to them.

---

## 1. The Manifold Hypothesis

Single-cell RNA sequencing measures the expression of tens of thousands of genes per cell, producing a data point in a space of dimension $D \sim 10^4$. Most of this space is empty: biologically meaningful states — cell types, differentiation trajectories, cell-cycle phases — live along smooth, low-dimensional paths or surfaces.

This is the **manifold hypothesis**: the data lie (approximately) on a smooth $d$-dimensional manifold $\mathcal{M}$ embedded in the ambient $\mathbb{R}^D$, with intrinsic dimension $d \ll D$.

Think of it this way: a spiral of cells differentiating from stem to mature state traces a 1-dimensional curve through thousands of gene dimensions. The biologically informative structure is the curve, not the 10,000-dimensional box it sits in.

The challenge for any analysis method is to *respect this intrinsic geometry* rather than imposing the geometry of the ambient box.

---

## 2. Why PCA Falls Short

The standard approach — PCA followed by kNN graph construction — projects data onto the directions of maximum *global* variance in gene space. This is mathematically equivalent to finding the best-fitting flat (hyperplane) subspace.

The problem: **curved manifolds cannot be faithfully represented by flat subspaces**. A spiral cannot be unrolled by projecting onto a plane without distorting distances along the spiral. 

More precisely: for PCA to preserve the geometry of $\mathcal{M}$, the tangent plane to $\mathcal{M}$ at *every* data point would need to point in the same direction — but on a curved manifold, tangent planes rotate as you move along the surface. A single global projection cannot align with all of them simultaneously.

This creates two compounding problems:

1. **Geometric distortion**: Euclidean distances in PCA space are poor proxies for actual distances along the manifold. Nearby cells on the manifold may appear far apart (or vice versa) after projection.
2. **Dropped signal**: Rare cell types and subtle trajectories often vary in directions of *low* global variance. PCA discards exactly these directions, making rare populations invisible.

TopoMetry addresses both problems by working directly with the intrinsic geometry of the data manifold, using operators derived from the **Laplace-Beltrami Operator**.

---

## 3. The Laplace-Beltrami Operator

On a curved surface $\mathcal{M}$ equipped with a Riemannian metric $g$ (a smoothly varying notion of distance and angle at each point), the **Laplace-Beltrami operator** (LBO) is defined as:

$$
\Delta_\mathcal{M} f = \frac{1}{\sqrt{|g|}}\, \partial_i\!\left(\sqrt{|g|}\; g^{ij}\, \partial_j f\right)
$$

In plain language: it is the natural generalisation of the ordinary second derivative to curved spaces. On flat Euclidean space it reduces to the familiar Laplacian $\sum_i \partial^2 f / \partial x_i^2$.

The LBO is self-adjoint and non-positive. Its eigenvalue equation is written:

$$
-\Delta_\mathcal{M}\, \varphi_k = \tilde\lambda_k\, \varphi_k, \qquad \tilde\lambda_k \geq 0
$$

with $0 = \tilde\lambda_0 \leq \tilde\lambda_1 \leq \tilde\lambda_2 \leq \cdots \to \infty$ (non-negative, increasing). The eigenfunctions $\{\varphi_k\}$ form a complete orthonormal basis: every function on $\mathcal{M}$ can be expanded as $f = \sum_k \langle f, \varphi_k\rangle \varphi_k$, exactly as Fourier modes on a circle. Small $\tilde\lambda_k$ corresponds to slowly-varying (smooth, global-scale) eigenfunctions; large $\tilde\lambda_k$ to rapidly-oscillating (fine-scale) ones.

**Key property**: The LBO spectrum encodes the full intrinsic geometry of $\mathcal{M}$ in a coordinate-free way. Two geometrically identical (isometric) manifolds have exactly the same spectrum $\{\tilde\lambda_k\}$ — the eigenfunctions are canonical descriptors of the surface's shape, independent of how it sits in ambient space.

> **Important**: TopoMetry does not directly decompose the LBO. Instead, it decomposes a closely related **diffusion operator** (see Section 7) whose eigenvalues live in $(0, 1]$ and are ordered *decreasingly* — the opposite of the LBO convention. The relationship is $\lambda_k \approx e^{-\tilde\lambda_k \varepsilon}$, where $\varepsilon$ is the kernel bandwidth.

---

## 4. Heat Diffusion: The Physical Intuition

The LBO governs how heat spreads on a surface. If you place a hot spot at point $y$ at time $t=0$ and let the heat diffuse, the temperature at point $x$ at time $t$ is the **heat kernel**:

$$
k_t(x, y) = \sum_{k=0}^\infty e^{-\tilde\lambda_k t}\, \varphi_k(x)\, \varphi_k(y)
$$

Notice: since $\tilde\lambda_k \geq 0$, each term decays in time. High-frequency eigenmodes (large $\tilde\lambda_k$) decay rapidly — fine-scale details wash out quickly. Low-frequency modes (small $\tilde\lambda_k$) persist — they capture coarse global shape.

At short times ($t \to 0$), Varadhan's theorem shows that the heat kernel approximates a Gaussian in *geodesic* distance:

$$
k_t(x,y) \approx \frac{e^{-d_\mathcal{M}(x,y)^2/4t}}{(4\pi t)^{d/2}}
$$

where $d_\mathcal{M}(x,y)$ is the distance *along* the manifold, not through ambient space. This means heat diffusion naturally recovers true geodesic distances.

A profound result (Bérard, Besson & Gallot, 1994) states that the map

$$
x \mapsto \left(e^{-\tilde\lambda_1 t/2}\varphi_1(x),\; e^{-\tilde\lambda_2 t/2}\varphi_2(x),\; \ldots\right)
$$

is an **isometric embedding** of $\mathcal{M}$: squared distances in this infinite-dimensional representation converge to geodesic distances squared as $t \to 0$. This is the rigorous theoretical foundation for Diffusion Maps — and for why diffusing on a cell graph is geometrically meaningful.

---

## 5. From Continuous to Discrete: Graph Laplacians and the Density-Bias Problem

In practice, we only have a finite point cloud sampled from $\mathcal{M}$. A fundamental theorem (Hein 2007; Belkin & Niyogi 2008) shows that the unnormalized graph Laplacian $L = D - W$ (with Gaussian-kernel edge weights) converges not to $\Delta_\mathcal{M}$, but to a *density-weighted* version:

$$
\frac{1}{nh^{d/2+1}} L f(x_i) \;\xrightarrow{n\to\infty}\; C_d\!\left[\Delta_\mathcal{M} f + g(\nabla \log p,\, \nabla f)\right]
$$

The extra term $g(\nabla \log p, \nabla f)$ is a **drift towards high-density regions**: the random walk is pulled toward wherever there are more cells, regardless of geometry. In single-cell data, $p(x)$ reflects both true biology and artefacts (sequencing depth, cell cycle, donor variation). The standard pipeline — kNN graph without density correction — therefore encodes a mixture of geometry and sampling statistics rather than pure manifold structure (in addition to the distortions introduced by PCA).

---

## 6. Removing the Density Bias: Two Approaches

TopoMetry implements two complementary strategies to recover the pure LBO.

### 6a. Diffusion Maps with $\alpha$-Normalisation

Coifman & Lafon (2006) showed that dividing the kernel by a power $\alpha$ of an empirical density estimate cancels the drift. First, estimate the local density:

$$
d_\varepsilon(x_i) = \sum_j \exp\!\left(-\frac{\|x_i - x_j\|^2}{\varepsilon}\right) \;\approx\; n\,\varepsilon^{d/2}(4\pi)^{d/2}\, p(x_i)
$$

Then form the $\alpha$-renormalised kernel:

$$
k_\varepsilon^{(\alpha)}(x_i, x_j) = \frac{k_\varepsilon(x_i, x_j)}{d_\varepsilon(x_i)^\alpha \cdot d_\varepsilon(x_j)^\alpha}
$$

and normalise rows to obtain a Markov (row-stochastic) transition matrix $T$. The limiting operator depends on $\alpha$:

| $\alpha$ | Limiting operator                                   | Meaning                                          |
| ---------- | --------------------------------------------------- | ------------------------------------------------ |
| $0$      | $\Delta_\mathcal{M} + 2g(\nabla\!\log p, \nabla)$ | Density maximally distorts geometry              |
| $1/2$    | $\Delta_\mathcal{M} + g(\nabla\!\log p, \nabla)$  | Partial correction                               |
| $1$      | $\Delta_\mathcal{M}$                              | **Pure geometry — density fully removed** |

Setting $\alpha = 1$ completely cancels the drift term. TopoMetry's default `bw_adaptive` kernel applies an adaptive local bandwidth (estimated from $k$-NN distances) *before* the $\alpha = 1$ renormalisation — a double correction that removes both global and local density effects.

### 6b. Continuous $k$-NN (CkNN) Graphs

An alternative approach (Berry & Sauer, 2019) corrects for density *by construction*. The key observation is that the distance to the $k$-th nearest neighbour scales as $\rho_k(x) \sim C\, p(x)^{-1/d}$ — small in dense regions, large in sparse ones. Normalising pairwise distances by the geometric mean of local neighbour distances factors out the density:

$$
A_{ij}^{\text{CkNN}} = \frac{\|x_i - x_j\|}{\delta\,\sqrt{\rho_k(x_i)\,\rho_k(x_j)}}
$$

Two points are connected (with weight 1) if $A_{ij}^{\text{CkNN}} < 1$. The **unnormalised** Laplacian of this binary graph is the *unique* unweighted $k$-NN-based construction that converges to the pure LBO.

---

## 7. The Diffusion Operator and Its Eigenvalues

After building a density-corrected similarity graph, TopoMetry constructs the **diffusion operator** — the row-stochastic Markov matrix:

$$
T_{ij} = \frac{k_\varepsilon^{(\alpha)}(x_i, x_j)}{\sum_j k_\varepsilon^{(\alpha)}(x_i, x_j)}
$$

$T$ describes a random walk on the cell graph: entry $T_{ij}$ is the probability of moving from cell $i$ to cell $j$ in one step. Its eigenpairs $\{(\lambda_k, \psi_k)\}$ satisfy $T\, \psi_k = \lambda_k\, \psi_k$.

**TopoMetry's eigenvalues live in $(0, 1]$ and are stored in decreasing order:**

$$
1 = \lambda_0 \geq \lambda_1 \geq \lambda_2 \geq \cdots \geq 0
$$

This is the **opposite** of the LBO convention. The connection is:

$$
\lambda_k \approx e^{-\tilde\lambda_k \varepsilon}
$$

where $\tilde\lambda_k \geq 0$ are the increasing LBO eigenvalues and $\varepsilon$ is the kernel bandwidth. A diffusion eigenvalue near 1 means the corresponding mode is smooth and geometrically persistent; a diffusion eigenvalue near 0 means the mode is noisy and rapidly-decaying.

The trivial eigenvalue $\lambda_0 = 1$ (the constant eigenfunction, meaning every cell looks the same at all scales) is always discarded (`drop_first=True`). The eigenspectrum (`tg.eigenspectrum()`) plots $\lambda_1 \geq \lambda_2 \geq \cdots$ — a decreasing curve that "elbows" toward zero as modes transition from signal to noise.

> **Floating-point note**: At 64-bit double precision, theoretically zero eigenvalues may be computed as small negative numbers (e.g., $-10^{-16}$). TopoMetry's msDM weighting uses $\lambda_k / (1 - \lambda_k)$, which diverges for $\lambda_k \to 1^-$ and is undefined for $\lambda_k \leq 0$. The code therefore uses only components satisfying $\lambda_k > 0$, making the pipeline numerically robust to this floating-point artefact.

### Contrast with Laplacian Eigenmaps (LE)

When the `method='LE'` option is used, TopoMetry instead decomposes the graph Laplacian $L = D - W$ and seeks its *smallest* non-zero eigenvalues $\tilde\lambda_1 \leq \tilde\lambda_2 \leq \cdots$ (increasing from 0). The eigenvalue ordering and scale are completely different from DM/msDM — LE eigenvalues are non-negative and unbounded, while DM eigenvalues are in $(0,1]$. The two are related by $\lambda_k \approx e^{-\tilde\lambda_k \varepsilon}$ but should not be compared directly.

---

## 8. The Spectral Scaffold: DM and msDM

The eigenvectors $\{\psi_k\}$ of $T$ converge to the eigenfunctions of $\Delta_\mathcal{M}$ — the "Fourier modes" of the cell-state manifold, ordered from smoothest ($\psi_1$, encoding global structure) to most oscillatory ($\psi_m$, encoding fine-scale detail).

### Diffusion Maps (DM)

The **Diffusion Map** embedding at diffusion time $t$ weights each eigenmode by $\lambda_k^t$:

$$
\Psi_t(x_i) = \bigl(\lambda_1^t\,\psi_1(x_i),\;\lambda_2^t\,\psi_2(x_i),\;\ldots,\;\lambda_m^t\,\psi_m(x_i)\bigr) \in \mathbb{R}^m
$$

Since $\lambda_k \in (0,1)$, raising to the power $t > 1$ suppresses small (noisy) eigenvalues more aggressively than large (geometrically meaningful) ones. The squared Euclidean distance in this space equals the **diffusion distance** — robust to noise because it integrates over all $t$-step paths simultaneously, not just the shortest path.

TopoMetry stores the DM scaffold as `X_spectral_scaffold` in `adata.obsm`.

### Multiscale Diffusion Maps (msDM)

For datasets with hierarchical structure (e.g., a cell-type tree with both broad lineages and fine sub-populations), a fixed diffusion time $t$ captures only one geometric scale. The **multiscale Diffusion Map** aggregates over all $t \geq 1$ via the geometric series $\sum_{t=1}^\infty \lambda^t = \lambda/(1-\lambda)$:

$$
\Psi_{\text{ms}}(x_i) = \left(\frac{\lambda_k}{1 - \lambda_k}\,\psi_k(x_i)\right)_{k=1}^m, \qquad \lambda_k > 0
$$

The weight $\lambda_k / (1-\lambda_k)$ is monotone increasing in $\lambda_k$: eigenmodes with large eigenvalues (globally-persistent geometry) are upweighted; eigenmodes with small eigenvalues (local fluctuations) are downweighted. This simultaneously encodes structure at all scales in a single representation.

TopoMetry uses the msDM scaffold as its default high-dimensional coordinate system, stored as `X_ms_spectral_scaffold` in `adata.obsm`.

### Reading the Eigenspectrum

The `tg.eigenspectrum()` scree plot shows $\lambda_1 \geq \lambda_2 \geq \cdots$ in decreasing order. How to interpret it:

- **Sharp initial drop, then plateau near 0**: a few dominant geometric modes capture most structure; the "elbow" suggests a natural intrinsic dimensionality.
- **Slow, gradual decay**: rich multi-scale structure with no single dominant scale — msDM is especially beneficial.
- **Gap between $\lambda_m$ and $\lambda_{m+1}$**: the first $m$ eigenmodes are well-separated from noise. TopoMetry's automated scaffold sizing uses this gap alongside MLE and FSA intrinsic-dimensionality estimators.

---

## 9. Measuring Distortion: The Riemannian Metric

Any reduction of high-dimensional data to 2D for visualisation *must* distort distances — it is geometrically impossible to perfectly flatten a curved surface. TopoMetry makes this distortion *visible and quantifiable*.

A smooth embedding $f: \mathcal{M} \to \mathbb{R}^N$ induces a **pullback metric** on $\mathcal{M}$:

$$
(f^* g_{\text{Eucl}})_{ij}(p) = \left(J_f^T J_f\right)_{ij}
$$

where $J_f$ is the Jacobian of $f$. If the embedding is isometric, all singular values $\sigma_k$ of $J_f$ equal 1. Deviations measure distortion:

- **Contraction** ($\sigma_k < 1$): the embedding compresses distances — cells that are far on the manifold appear close in the plot.
- **Expansion** ($\sigma_k > 1$): the embedding stretches distances — cells that are close on the manifold appear far in the plot.
- **Anisotropy** ($\sigma_1/\sigma_d \gg 1$): different directions are distorted by different amounts — trajectories are bent or twisted.

The `tp.sc.plot_riemann_diagnostics()` and `tp.sc.calculate_deformation_on_projection()` functions compute these local distortion fields and overlay them on 2D embeddings, turning a qualitative visualisation into a quantitatively interpretable map.

---

## 10. TopoMetry's Pipeline: Putting It All Together

```
Raw expression matrix (n cells × D genes)
        │
        ▼
  Normalise + HVG selection + Z-score scaling
        │
        ▼
  1. kNN graph in HVG space  (no PCA)
        │
        ▼
  2. Density-corrected diffusion operator T  [bw_adaptive: local bandwidth + α=1]
     Eigenvalues: 1 ≥ λ₁ ≥ λ₂ ≥ … ≥ 0  (decreasing, in (0,1])
        │
        ▼
  3. Eigendecomposition → DM scaffold  (ψₖ × λₖᵗ)
                        → msDM scaffold (ψₖ × λₖ/(1−λₖ))
     [automated sizing via intrinsic dimensionality + spectral gap]
        │
        ▼
  4. kNN graph in scaffold space
        │
        ▼
  5. Refined diffusion operator on scaffold → P(Z), P(msZ)
        │
        ▼
  6. Graph-layout optimisation (MAP / PaCMAP) → 2D visualisation
        │
        ▼
  7. Evaluation: geometry-preservation scores and distortion quantification via Riemannian metric
```

**Step 1 (kNN in HVG space, not PC space)** avoids the metric distortion introduced by PCA. After variance-stabilising normalisation, Euclidean distances in HVG space are already a reasonable proxy for geodesic distances — unlike distances in PCA space, which are distorted by curvature and by directions discarded by the projection.

**Steps 2–3** implement the density-corrected LBO approximation. The eigenvectors of the resulting diffusion operator converge to the LBO eigenfunctions on the cell-state manifold. Critically, the eigenvalues are in $(0,1]$ and decreasing — large values mark geometrically meaningful modes, small values near 0 mark noise.

**Step 5** is a second round of diffusion on the scaffold coordinates, further smoothing residual noise while preserving the structure captured in the first decomposition.

**Step 7** lets you see *where* your 2D plot is trustworthy and where it is deceptive.

---

## 11. Summary: Why This Is Better

| Property              | PCA-based pipelines (Seurat, Scanpy default)                                | TopoMetry                                                      |
| --------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Geometry model        | Global linear (flat)                                                        | Local nonlinear (curved manifold)                              |
| Density bias          | Present ($\alpha=0$ → converges to $\Delta_p \neq \Delta_\mathcal{M}$) | Removed ($\alpha=1$ or CkNN)                                 |
| Eigenvalue type       | Variance fractions (no diffusion time)                                      | Diffusion operator:$1 \geq \lambda_1 \geq \cdots \geq 0$     |
| Scale sensitivity     | Single scale (fixed$k$ PCs)                                               | Multi-scale (msDM:$\lambda/(1-\lambda)$ sums over all $t$) |
| Rare-cell detection   | May miss low-variance subtypes                                              | Preserved (no global variance filter)                          |
| Distortion visible?   | No                                                                          | Yes (Riemannian metric diagnostics)                            |
| Theoretical guarantee | None (heuristic)                                                            | Convergence to LBO eigenfunctions                              |

In brief: TopoMetry replaces a heuristic pipeline built for computational convenience with one that is mathematically justified to recover the intrinsic geometry of the cell-state manifold. The practical consequence is better separation of rare populations, more faithful trajectories, and visualisations where distortion is explicit rather than hidden.

---

## Further Reading

The mathematical foundations are detailed in:

- Coifman & Lafon (2006) — Diffusion Maps and the $\alpha$-normalisation.
- Coifman & Maggioni (2006) — Multiscale Diffusion Maps.
- Berry & Sauer (2019) — CkNN and the unique unweighted LBO-consistent graph.
- Bérard, Besson & Gallot (1994) — The heat kernel isometric embedding theorem.
- Hein, Audibert & von Luxburg (2007) — Convergence of graph Laplacians to the LBO.

For a self-contained mathematical treatment covering all of the above in a single-cell context, see the [preprint](https://github.com/davisidarta/topometry).
