# Why TopoMetry? Why never PCA?

## Single-cell data has geometry

When you sequence thousands of cells, each one becomes a point in a space with
as many dimensions as there are measured features — typically thousands of genes.
That sounds abstract, but it encodes something concrete: cells of the same type
cluster together, cells undergoing differentiation sit along continuous trajectories,
and cells at intermediate states fall between defined endpoints. The data is not a
random cloud. It has structure, and that structure is biology.

This structure is what mathematicians call a *manifold*: a curved, lower-dimensional
surface embedded in the high-dimensional feature space. A progenitor cell differentiating
into two distinct fates traces a branching path along this manifold. A cell cycling
through G1, S, and G2/M traces a closed loop. Activation states, effector gradients,
and clonal imprints are all geometric features of this surface. Understanding
single-cell data means understanding this shape — not just the directions along which
cells vary the most.

## Why PCA fails: the intuitive case

Principal Component Analysis (PCA) is a linear method. It finds the directions in
gene expression space along which cells vary the most, rotates the coordinate system
to align with those directions, and discards the rest. This is a sensible strategy
when data genuinely forms a linear structure — a cloud elongated in a few directions
and flat in others.

Single-cell data is rarely like this. Consider the cell cycle: cells progress from G1
to S to G2/M and back, tracing a loop in gene expression space. A single linear
component cannot represent a loop — a one-dimensional projection of a circle is
always an interval, with the two ends of the trajectory mapped to the same region
regardless of how many intermediate states separate them. In high-dimensional data,
where the cell cycle signal competes with many other sources of variation, this
distortion is compounded: the circular structure gets fragmented across multiple
principal components rather than cleanly recovered in any one of them. The result,
visible in the paper's benchmarks, is that mitotic cells are misplaced along
differentiation axes rather than positioned alongside other cycling populations where
they belong.

The same logic applies to any nonlinear structure: differentiation branches require
preserving a fork that no linear subspace can encode faithfully; continuous activation
gradients that curve through expression space get projected onto straight axes that
miss the curvature; rare populations that occupy geometrically distinct but
low-variance corners of the manifold are compressed toward the bulk.

An analogy: take a sheet of paper and roll it into a cylinder. Its two largest PCA
directions are the length axis and a cross-sectional axis — neither captures the
fact that the sheet is rolled. Unrolling it is what exposes the true structure.
TopoMetry is, in spirit, the unrolling step.

## Why PCA fails: the mathematical case

PCA computes the top-$k$ eigenvectors of the sample covariance matrix

$$
C = \frac{1}{N} \sum_{i=1}^{N} x_i x_i^\top \in \mathbb{R}^{D \times D}
$$

after centering. The fraction of total variance explained by $k$ components is

$$
\mathrm{EVR}_k = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{D} \lambda_j}
$$

where $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D \geq 0$ are the eigenvalues
of $C$. The implicit assumption is that a small number of eigenvalues dominate — that
the spectrum has a clean gap after the first $k$ components, justifying discarding the
rest. This assumption holds for genuinely linear data. It breaks for nonlinear
manifolds.

A clean, exact example illustrates why. Consider $N$ points drawn uniformly from the
unit circle $S^1 \subset \mathbb{R}^2$. This data has intrinsic dimension $d = 1$,
yet its population covariance is

$$
C = \mathbb{E}[xx^\top] = \begin{pmatrix} \mathbb{E}[\cos^2\theta] & \mathbb{E}[\cos\theta\sin\theta] \\ \mathbb{E}[\cos\theta\sin\theta] & \mathbb{E}[\sin^2\theta] \end{pmatrix} = \frac{1}{2} I_2
$$

Both eigenvalues are identical: $\lambda_1 = \lambda_2 = \frac{1}{2}$. PCA assigns
each coordinate direction exactly equal importance and finds no preferred projection
— it is completely uninformative about the circular structure, and cannot distinguish
the 1-dimensional ring from 2-dimensional isotropic noise. For this particular
example, the failure is exact and total.

The circle is a symmetric, highly regular object, and its perfectly flat spectrum is
a consequence of that symmetry. Real single-cell manifolds are neither symmetric nor
regular, so one should not expect the eigenvalues to be exactly equal in practice.
The broader point is different: as the geometry of data becomes more nonlinear and
curved, the covariance spectrum tends to flatten, EVR decreases, and no small set of
principal components can faithfully represent the structure — not because all
eigenvalues are equal, but because the variance is distributed across many directions
in a way that reflects the data's intrinsic curvature rather than any meaningful
low-rank linear approximation.

Real single-cell data exhibits exactly this signature. Across a benchmark of 68
scRNA-seq datasets, PCA typically explains around 36% of total variance — sometimes
as little as 20% — and retaining more components does not recover the rest. The
eigenspectrum genuinely flattens after 30–50 components; there is no hidden gap
waiting to be found. In the machine learning literature, slowly decaying covariance
spectra are a well-established hallmark of nonlinear data. In single-cell genomics,
this has gone largely unexamined, despite the fact that every major analysis step
depends on PCA as its first operation.

The practical consequence: the neighborhood graph built in PCA space is built on a
corrupted metric. Every downstream step — clustering, visualization, pseudotime,
trajectory inference — inherits that corruption.

## The sampling problem

Even setting aside nonlinearity, PCA faces a second structural problem: it weights
all cells equally regardless of how frequently each cell type was sampled. The total
covariance matrix decomposes as

$$
C_{\text{total}} = \underbrace{\sum_k \frac{n_k}{N} C_k}_{\text{within-group}} + \underbrace{\sum_k \frac{n_k}{N} (\mu_k - \mu)(\mu_k - \mu)^\top}_{\text{between-group}}
$$

where $C_k$ is the within-population covariance, $\mu_k$ is the mean expression
vector of population $k$, $\mu$ is the global mean, and $n_k / N$ is the abundance
of population $k$. The between-group term encodes how far each population's centroid
sits from the global mean, weighted by its abundance. This is what drives PCA's
ability to separate transcriptionally distinct cell types at all — a rare population
with a highly distinctive expression profile can still generate a large between-group
contribution and appear in the top principal components.

The within-group term, however, tells a different story. It is a weighted average of
each population's internal covariance structure, with weights proportional to cell
abundance. This means that the local geometry of rare populations — the internal
variation that distinguishes their subtypes, activation states, or clonal structure —
contributes negligibly to the top principal components relative to the internal
variation of common populations. PCA is not blind to rare cell types when they are
transcriptionally far from the rest of the data, but it systematically underrepresents
their internal structure and compresses their local neighborhoods. It is precisely
this internal geometry that encodes the most biologically interesting variation within
a population — and that TopoMetry is designed to recover.

This is not a problem that can be fixed by selecting more highly variable genes; in
fact, increasing the number of features tends to make it worse, as each additional
gene introduces more nonlinear variation that PCA cannot represent. The issue is
structural.

## What TopoMetry does instead

Rather than projecting cells onto linear axes of maximum variance, TopoMetry asks:
given the local neighborhood of each cell, what is the geometry of the space it
inhabits?

It starts by building a cell–cell similarity graph with kernels that adapt to the
local sampling density and intrinsic dimensionality of each cell's neighborhood. A
cell in a densely sampled region and a cell in a sparse region are treated
differently: the kernel bandwidth scales with the local neighborhood radius, so the
resulting similarity measure reflects local geometry rather than global density. This
directly addresses the sampling bias that PCA-based graphs accumulate.

From this graph, TopoMetry approximates the Laplace–Beltrami operator (LBO), the
natural generalization of the Laplacian to curved spaces. On a smooth Riemannian
manifold $(\mathcal{M}, g)$, the LBO $\Delta_g$ admits an eigendecomposition

$$
\Delta_g \phi_\ell = \mu_\ell \phi_\ell, \quad 0 = \mu_1 \leq \mu_2 \leq \cdots
$$

whose eigenfunctions $\phi_1, \phi_2, \ldots$ form a complete orthonormal basis of
$L^2(\mathcal{M})$ — a Fourier basis intrinsic to the manifold. The trivial first
eigenfunction ($\mu_1 = 0$, corresponding to the constant function on each connected
component) is dropped from embeddings. The remaining eigenfunctions are ordered by
the scale of variation they capture: low-frequency eigenfunctions capture the broadest
organization of the data — separation between major lineages, large-scale
developmental trajectories — while higher-frequency eigenfunctions resolve
progressively finer structure such as local activation gradients, cell cycle phases,
or clonal imprints. Crucially, these coordinates are intrinsic to the manifold's
geometry: they do not depend on how the manifold happens to be embedded in the
ambient gene expression space, and they are not biased by the global variance of any
particular direction.

TopoMetry's spectral scaffold is a data-driven approximation of this eigenbasis,
computed from the graph Laplacian built on the adaptive similarity graph. Under
appropriate conditions on the kernel and sampling density, the graph Laplacian
converges to $\Delta_g$, making the scaffold's eigenvectors faithful intrinsic
coordinates of the data manifold. The number of retained components is determined
automatically from intrinsic dimensionality estimates rather than fixed to an
arbitrary number.

A multiscale scaffold then aggregates coordinates across all diffusion timescales
analytically, blending local neighborhood structure and long-range organization into
a single compact representation. A refined graph built in this scaffold space captures
the geometry of the geometry and serves as the backbone for clustering, visualization,
trajectory inference, and all other downstream analyses.

## Evaluating representations

TopoMetry does not simply produce an embedding and ask you to trust it. It includes
operator-native metrics that compare diffusion operators at local, mesoscopic, and
global scales, giving a quantitative measure of how faithfully any representation
preserves the original data geometry. It also provides Riemannian distortion
diagnostics — tools that reveal where a 2-D embedding expands, contracts, or shears
the underlying manifold, so that visual interpretations can be grounded in an
understanding of the distortions they introduce.

This matters because visually coherent embeddings can be geometrically misleading,
and there is currently no standard practice in the field for checking this. TopoMetry
makes evaluation a first-class step in the analysis rather than an afterthought.
