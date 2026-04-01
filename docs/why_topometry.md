# Why geometry? Why not variance?

David Sidarta Oliveira, University of Oxford, 2026


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

Principal Component Analysis (PCA) is a linear method originally introduced by
Pearson ([1901](https://doi.org/10.1080/14786440109462720)) and later formalized as
a general statistical tool by Jolliffe ([1986](https://doi.org/10.1007/978-1-4757-1904-8)).
It finds the directions in gene expression space along which cells vary the most,
rotates the coordinate system to align with those directions, and discards the rest. This is a sensible strategy
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

## How PCA became the standard — and why no one checked the math

Given the problems outlined above, a natural question is how PCA became so deeply
embedded in single-cell analysis in the first place. The answer is not that the field
carefully evaluated it and found it adequate; it is that PCA entered the workflow for
practical reasons at a specific historical moment, and the practice propagated by
continuity rather than by critical assessment.

In science, it is often easier and faster to adopt published practice than to
construct and justify a new one, especially in a field of biological rather than
computational focus. That is precisely what happened in single-cell genomics: the
widespread use of PCA as a preprocessing step did not arise from mathematically
grounded observations about single-cell data, but was inherited from adjacent
computational practice and then frozen into infrastructure.

### The t-SNE origin

The immediate precursor was t-distributed stochastic neighbor embedding (t-SNE), the
visualization algorithm that dominated single-cell data exploration before UMAP. In
the original t-SNE paper, van der Maaten and Hinton explicitly recommended reducing
data to 30 dimensions with PCA before running t-SNE:

> "Note that for large data sets, it may be necessary to use a reduced representation
> of the data. We typically reduce the data to 30 dimensions using PCA."
>
> — van der Maaten & Hinton (2008), *Visualizing Data using t-SNE*,
> [J. Mach. Learn. Res. 9, 2579–2605](https://www.jmlr.org/papers/v9/vandermaaten08a.html)

The reason was purely computational. Early t-SNE implementations computed pairwise
affinities in the full feature space, which scales quadratically in both the number
of cells and the number of features. For the datasets available at the time — in many
cases not biological at all, or far smaller than even early scRNA-seq experiments —
this was already intractable on most hardware. PCA to 30 dimensions was a pragmatic
compression step that made t-SNE feasible, with no claim that 30 PCs faithfully
captured the data geometry. The recommendation was a workaround, not a principled
preprocessing strategy.

### Seurat, scran, and the packaging of PCA as a default

When single-cell RNA-seq datasets began scaling into the tens of thousands of cells,
the first comprehensive analysis toolkits — Seurat
([Satija et al., 2015](https://doi.org/10.1038/nbt.3192)) and scran
([Lun et al., 2016](https://doi.org/10.12688/f1000research.9501.2)) — needed to
provide accessible end-to-end workflows for a biological audience. Both adopted the
same reasoning: sacrifice representational fidelity for computational speed and
accessibility, on the assumption that the cost would be minimal. Seurat in particular
packaged PCA → kNN graph → Leiden clustering → UMAP or t-SNE into a handful of
function calls, making it straightforward for users without a computational background
to produce publication-ready figures from raw count matrices.

The reasoning was understandable in context. Early scRNA-seq datasets were small by
today's standards — hundreds to a few thousand cells — and the transcriptional
differences between major cell types were large enough that PCA's distortions did not
prevent correct annotation of the most abundant populations. The true cost of PCA
was invisible precisely because the analyses it was most likely to fail on —
fine-grained subpopulation structure, rare populations, continuous gradients — were
not accessible from the data volumes available at the time. By the time datasets grew
large enough for these failures to become visible, the PCA-based workflow had already
been institutionalized.

### Scanpy and field-wide adoption

Scanpy ([Wolf et al., 2018](https://doi.org/10.1186/s13059-017-1382-0)) and the
broader scverse ecosystem followed the same design choices and made them the default
for a much larger user base. With the publication of widely read "best practices"
guides — most notably Luecken and Theis (2019)
([Mol. Syst. Biol. 15, e8746](https://doi.org/10.15252/msb.20188746)) and its
successor Heumos et al. (2023)
([Nat. Rev. Genet. 24, 550–572](https://doi.org/10.1038/s41576-023-00586-w)) —
the PCA-based pipeline was codified as the recommended approach across modalities and
experimental contexts. These guides introduced millions of new single-cell analysis
users to a workflow in which PCA is presented as an unexamined first step, not as a
methodological choice with known limitations and alternatives.

### What users are shown — and what they are not

Both Seurat and Scanpy provide diagnostic tools intended to help users choose how many
principal components to retain. In Seurat, the standard tool is `ElbowPlot()`, which
plots $\sigma_k = \sqrt{\lambda_k}$ — the standard deviation, not the variance — of
each principal component on the $y$-axis against component rank on the $x$-axis.
Scanpy's `sc.pl.pca_variance_ratio()` plots the per-component fraction of variance
$\lambda_k / \sum_j \lambda_j$ on a logarithmic $y$-axis. In both cases, users are
asked to identify a visual "elbow" — an informal subjective judgment with no
established statistical definition — and to retain all components up to that point.

Neither plot shows the quantity that actually matters: the cumulative explained
variance ratio

$$
\mathrm{EVR}_k^{\text{cumul}} = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{D} \lambda_j}
$$

which is the only number that directly answers the question "what fraction of the
total variance in the data is captured by my PCA representation?" A user inspecting
an elbow plot cannot determine whether 20 PCs explain 30% of the total variance or
85%; the shape of the per-component curve does not convey this. The logarithmic
$y$-axis used in Scanpy further compresses differences between components, making a
genuinely flat spectrum — the signature of a nonlinear system, as discussed above —
visually indistinguishable from a spectrum with a meaningful gap. Seurat additionally
offers the JackStraw test
([Macosko et al., 2015](https://doi.org/10.1016/j.cell.2015.05.002)),
which assesses whether each PC explains significantly more variance than expected by
chance under permutation. This tests whether a component's eigenvalue is
distinguishable from noise, but it says nothing about whether the retained components
collectively capture a meaningful fraction of the biological signal. A PC can pass the
JackStraw test and still represent 0.3% of total variance.

The result is that users routinely select 20–50 PCs on the basis of a visual
heuristic applied to a plot that does not show cumulative coverage, have no practical
way to know from standard software output that their representation may capture less
than 40% of the signal, and proceed to all downstream analyses — clustering,
differential expression, trajectory inference — with this information gap unexamined.

### A pattern of implicit reliance

Beyond the software defaults, there is a broader pattern in the literature of PCA
being applied without explicit acknowledgment in methods sections. This is
particularly common in manuscripts that present or use nonlinear dimensionality
reduction and trajectory methods, where the linear preprocessing step is an awkward
admission for tools positioned as geometry-aware. PHATE
([Moon et al., 2019](https://doi.org/10.1038/s41587-019-0336-3)) and related tools
from the same group apply PCA for initial dimensionality reduction in their
implementations; the degree to which this is disclosed prominently in methods sections
varies considerably across publications using these tools. The diffusion pseudotime
framework ([Haghverdi et al., 2016](https://doi.org/10.1038/nmeth.3971)) and several
subsequent trajectory methods from the same tradition build diffusion operators on
PCA-derived neighborhood graphs without always making this explicit in the description
of the method itself, where the focus is understandably on the nonlinear step. The
effect, across dozens of papers, is that readers absorb the impression that diffusion-
or graph-based methods operate on raw or minimally processed data, when in practice
they inherit the distortions introduced by PCA before any nonlinear computation
begins.

This is not confined to specific groups or tools; it is a field-wide pattern that
reflects how PCA became infrastructure: invisible, unquestioned, and therefore
unexamined. The benchmarking literature on single-cell methods
([Luecken et al., 2022](https://doi.org/10.1038/s41592-021-01336-8)) has compared
algorithms assuming PCA preprocessing as a shared starting point, which means that
the comparative evaluations themselves are built on a foundation whose adequacy has
not been tested. No study prior to TopoMetry systematically measured whether
PCA-based neighborhood graphs preserve the geometry of the data they are built on.
Until that measurement exists, the field has no principled basis for knowing how much
biological signal its standard workflow discards before any downstream analysis
begins.

## Variational methods: the same assumption, differently packaged

Methods such as scVI ([Lopez et al., 2018](https://doi.org/10.1038/s41592-018-0229-2)),
totalVI ([Gayoso et al., 2021](https://doi.org/10.1038/s41592-020-01050-x)), and
their relatives have become standard in single-cell genomics as apparent improvements
over PCA. They use deep neural networks to learn nonlinear encoders and decoders, and
they model count data with appropriate likelihood functions (typically negative
binomial). These are genuine advances. But at the core of their training objective
lies an assumption that is just as incompatible with single-cell geometry as PCA's
linearity — and it is one that rarely gets examined.

All of these methods are variational autoencoders (VAEs), a framework introduced by
Kingma & Welling ([2013](https://arxiv.org/abs/1312.6114)). They are trained by
maximizing the Evidence Lower BOund (ELBO):

$$
\mathcal{L}(\theta, \phi;\, x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right]}_{\text{reconstruction}} - \underbrace{\mathrm{KL}\!\left(q_\phi(z \mid x) \;\|\; p(z)\right)}_{\text{regularization}}
$$

where $q_\phi(z \mid x)$ is the approximate posterior (encoder), $p_\theta(x \mid z)$
is the likelihood (decoder), and $p(z)$ is the prior over the latent space. In every
widely used implementation of these methods for single-cell data, the prior is a
standard isotropic Gaussian: $p(z) = \mathcal{N}(0, I_d)$.

The KL regularization term penalizes the encoder for producing posteriors that deviate
from this prior. Its effect on the aggregate posterior — the distribution of latent
representations across all cells, $q_\phi^*(z) = \int q_\phi(z \mid x)\, p_{\mathrm{data}}(x)\, dx$
— is to push it toward $\mathcal{N}(0, I_d)$. This is not a side effect; it is by
design. The isotropic Gaussian prior is chosen precisely because it encourages a
well-organized, continuous latent space where nearby points in $\mathbb{R}^d$ decode
to similar gene expression profiles.

The problem is that $\mathcal{N}(0, I_d)$ is a very specific geometric object: it is
unimodal, isotropic, and supported on all of $\mathbb{R}^d$, which is contractible —
it has no holes, no branches, no disconnected components. Single-cell manifolds, by
contrast, can be any of these things. The cell cycle is topologically a loop. Multiple
independent lineages are topologically disconnected. A branching differentiation tree
has the topology of a graph with cycles removed. Forcing the aggregate posterior to
look like a Gaussian ball is topologically incompatible with the true data geometry.

Concretely: if the data lies on a manifold $\mathcal{M}$ with $k$ disconnected
components — say, $k$ distinct cell lineages — and the latent prior is $\mathcal{N}(0, I_d)$,
then the encoder must map cells from all $k$ components into a single connected,
unimodal distribution. Cells from different lineages will necessarily be pushed
together in the latent space to satisfy the prior. Rare populations are pulled hardest:
the reconstruction term weights each cell equally, so it favors fitting common cell
types well, while the KL term pulls the rare population's posterior toward the prior
regardless of where that pulls it relative to other populations. The end result is a
latent space whose local geometry reflects the prior distribution more than the data
manifold.

This is, at its core, a variance-based assumption in nonlinear disguise. The Gaussian
prior assigns probability proportional to $\exp(-\|z\|^2/2)$, which decays with
squared distance from the origin. Maximizing the ELBO subject to this prior
encourages latent codes to spread evenly across a ball of fixed radius in $\mathbb{R}^d$
— an implicit form of variance maximization in the latent space. The fundamental error
is the same as PCA's: a parametric distributional assumption that is inconsistent with
the topology and geometry of the data is used to define what a "good" representation
looks like. In PCA, that assumption is linearity. In VAEs, it is Gaussianity. Both
assumptions are convenient, both are mathematically tractable, and both are wrong for
single-cell data.

This is not a hypothetical concern. Supplementary Figure S1a of the TopoMetry
manuscript directly shows that scVI's latent representations fail to preserve data
geometry in benchmarks across the same 68 datasets where PCA fails — and for
consistent reasons. The Gaussian prior topology does not match the data topology, and
the latent space reflects that mismatch.

It is worth noting that this is an active area of research in the machine learning
community, where alternatives such as hyperspherical priors, Riemannian VAEs, and
topologically-aware latent spaces have been proposed precisely to address this
problem. None of these have been adopted in the standard single-cell toolbox. Until
they are, any method that relies on a Gaussian latent prior shares the core limitation
described here — regardless of how sophisticated its encoder or decoder architecture
is.

## What TopoMetry does instead

Rather than projecting cells onto linear axes of maximum variance, or fitting a
generative model that imposes a distributional prior on the latent space, TopoMetry
asks a more direct question: given the local neighborhood of each cell, what is the
geometry of the space it inhabits?

It starts by building a cell–cell similarity graph with kernels that adapt to the
local sampling density and intrinsic dimensionality of each cell's neighborhood. A
cell in a densely sampled region and a cell in a sparse region are treated
differently: the kernel bandwidth scales with the local neighborhood radius, so the
resulting similarity measure reflects local geometry rather than global density. This
directly addresses the sampling bias that both PCA-based graphs and VAE reconstruction
losses accumulate.

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
particular direction, nor constrained to match any prescribed distributional form.

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