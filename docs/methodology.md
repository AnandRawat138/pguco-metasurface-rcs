# Methodology: Physics-Guided Uncertainty-Aware Counterfactual Local Optimisation (PGUCO)

This document states exactly what the code computes. Every equation carries a label used in the
docstrings of the module that implements it; no equation appears here that the code does not use.
Where the implementation deliberately differs from a textbook form, the difference is stated.

**Research question.** How can an existing metasurface be improved using a very limited number of
additional high-fidelity electromagnetic simulations?

**Hypothesis under test.** A physics-aware, uncertainty-guided counterfactual optimiser improves an
existing design with fewer high-fidelity evaluations than conventional black-box optimisation.
The benchmark and ablation engines (`experiment.run_benchmark`, `evaluation/ablation.py`) exist to
test this; they do not presuppose the answer.

---

## 1. Problem formulation

Given an existing design $\mathbf{x}_0$ and its response $\mathbf{y}_0$ (simulated, measured or reported),
find

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}\in\mathcal{X}} \; J(\mathbf{x})
\quad\text{s.t.}\quad g_i(\mathbf{x}) \le 0,\; i=1..n_g,
$$

using at most $N_{\max}$ additional high-fidelity (HF) evaluations, or, in multi-objective mode, an
approximation of the Pareto set of $\tilde{\mathbf{J}}(\mathbf{x})$.

| Symbol | Meaning |
|---|---|
| $\mathbf{x}\in\mathbb{R}^d$ | design vector (schema order) |
| $\mathcal{X}=\prod_i[x_i^{\min},x_i^{\max}]$ | box defined by the parameter schema |
| $\mathbf{u}(\mathbf{x})$ | unit-cube coordinates $u_i=(x_i-x_i^{\min})/(x_i^{\max}-x_i^{\min})$ |
| $\mathcal{F}\subseteq\{1..d\}$ | free (non-fixed) parameters; $d_f=|\mathcal{F}|$ |
| $\mathbf{y}=f(\mathbf{x})$ | EM response produced by an HF source (never by a surrogate) |
| $\mathbf{J}=[J_1..J_m]$ | objective vector, every component minimised |
| $N_{\mathrm{HF}}$ | number of HF evaluations charged so far |
| $\delta_i$ | perturbation step of parameter $i$ (schema `step`) |

Implementation: `optimization/problem.py` (`OptimizationProblem`).

## 2. Design representation

A user-editable **parameter schema** (`geometry/parameter_schema.py`, YAML/JSON) lists every
variable with type (continuous, integer, categorical), bounds, unit, default, step $\delta_i$,
manufacturing resolution, fixed flag and semantic *role* (period, length, width, gap, rotation,
thickness, permittivity, loss tangent, ...). Optimisers act on $\mathbf{u}$; designs are snapped to the
manufacturing resolution before evaluation.

The schema names a **geometry template** (`geometry/geometry_generator.py`: rectangular patch,
two-state checkerboard super-cell, square loop, slotted patch, cross dipole; new templates are
registered with a decorator). A template produces a `CellGeometry`: shapely polygons for patterned
metal layers plus a layer stack. This parameterised geometry is the design of record; STL files are
derived exports.

Other inputs (`geometry/image_parser.py`, `geometry/stl_loader.py`, `geometry/mask_encoder.py`) yield
*approximate*, confidence-rated parameter estimates (HIGH/MEDIUM/LOW) and never claim a unique
inverse from a mesh or an image.

## 3. EM response representation

`electromagnetics/`: complex $S_{ij}(f)=S^R_{ij}(f)+jS^I_{ij}(f)$ keyed by port pair and polarisation
(`SParameterSet`; magnitude, phase, dB, real and imaginary views; Touchstone and CSV I/O;
magnitude-only data is kept separately and phase is never invented), RCS
$\sigma(f,\theta,\phi)$ in m² with $\sigma_{\mathrm{dBsm}}=10\log_{10}\sigma$ (`RCSData`: monostatic,
bistatic, specular, reference plate, TE/TM/any label), and far-field $E_\theta, E_\phi$ with
$\sigma=4\pi|rE_s|^2/|E_i|^2$ (`FarField`). Every response carries a provenance label
(`core/provenance.py`): PREDICTED, VALIDATED (full-wave), MEASURED, REPORTED,
REPORTED (plot-digitized), or SYNTHETIC ORACLE.

Angle convention: incidence from $(\theta_i,\phi_i)$; monostatic $=(\theta_i,\phi_i)$; specular
$=(\theta_i,\phi_i+180^\circ)$.

## 4. Objective functions (`optimization/objectives.py`)

All objectives are minimised. $\mathcal{B}=\{f: f_l\le f\le f_h\}$ is the configured band, $w_f$
frequency weights (default 1), $\mathcal{P}$ polarisations, $\Theta$ incidence angles, and
$s(\cdot)$ the configured scale: linear $\sigma$, $\sigma_{\mathrm{dBsm}}$, or reduction
$\rho=10\log_{10}(\sigma/\sigma_{\mathrm{ref}})$ relative to a reference (PEC plate of equal size).

- **O1 specular:** $J_{\mathrm{spec}}=\dfrac{\sum_{f\in\mathcal{B}} w_f\, s(\sigma_{\mathrm{spec}}(f))}{\sum_f w_f}$, averaged over the selected $(p,\theta)$ or their maximum (`aggregate: max`).
- **O2 broadband:** $J_{\mathrm{broad}}=\dfrac{\int_{f_l}^{f_h} w(f)\,s(\sigma(f))\,df}{\int_{f_l}^{f_h} w(f)\,df}$ (trapezoidal rule).
- **O3 angular robustness:** $J_{\mathrm{ang}}=\dfrac{\sum_{\theta} w_\theta\,\bar\sigma(\theta)}{\sum_\theta w_\theta}$ or $\max_\theta\bar\sigma(\theta)$, with $\bar\sigma(\theta)$ the band average (O1 form) averaged over polarisations.
- **O4 polarisation robustness:** $J_{\mathrm{pol}}=\frac{1}{N_p}\sum_p J_p$ or $\max_p J_p$, $J_p$ the band/angle average for polarisation $p$.
- **O5 diffusion:** $p_i=P_i/\sum_jP_j$ with $P_i=\sigma_i\,\Delta\Omega_i$ the power scattered into solid-angle cell $i$ ($\Delta\Omega_i=\sin\theta_i\Delta\theta\Delta\phi$); $H_{\mathrm{ang}}=-\sum_ip_i\log(p_i+\epsilon)$, optionally divided by $\log N$; $J_{\mathrm{diff}}=-\overline{H_{\mathrm{ang}}}$ over the band. Maximum entropy is *one* configurable notion of diffusion, not assumed to be the correct stealth objective.
- **O6 total scattered power:** band mean of $\frac{1}{4\pi}\int\sigma\,d\Omega$.
- **O7 reduction bandwidth:** $J_{\mathrm{bw}}=-\mathrm{FBW}$, the fractional bandwidth $2(f_2-f_1)/(f_2+f_1)$ of the largest contiguous band with $\rho\le\rho_{\mathrm{th}}$ (default $-10$ dB).
- **O8 S-parameter level:** band mean of $|S_{ij}|_{\mathrm{dB}}$ (negated if `sense: max`).
- **O9 manufacturing margin:** $J_{\mathrm{mfg}}=\max_i g_i(\mathbf{x})$ (design-only).

### Normalisation (N1–N3) and composite (C1)

Objectives are never combined without normalisation. With statistics frozen once:

- **N1 min-max:** $\tilde J_k=(J_k-J_k^{\min})/(J_k^{\max}-J_k^{\min})$ (fitted on the initial HF data)
- **N2 z-score:** $\tilde J_k=(J_k-\mu_k)/s_k$ (fitted on the initial HF data)
- **N3 baseline-relative (default):** $\tilde J_k=(J_k-J_{0,k})/\max(|J_{0,k}|,\varphi_k)$ with a per-objective floor $\varphi_k\ge0$ (`norm_floor`, default 0) that prevents division by a near-zero baseline value.

$$
\textbf{(C1)}\qquad J(\mathbf{x})=\sum_k\lambda_k\tilde J_k(\mathbf{x})+\lambda_c'\,J_{\mathrm{constraint}}(\mathbf{x}),
\qquad \lambda_c'=\lambda_c\,\lambda_g/10
$$

where $\lambda_k$ are objective weights, $\lambda_c$ the configured constraint weight and $\lambda_g$ the
physics-config geometry weight (so disabling geometry physics sets $\lambda_c'=0$). The method and
its fitted statistics are written to the experiment configuration. In multi-objective mode the
vector $\tilde{\mathbf{J}}$ is used instead of C1 (Sec. 12).

Relative improvement: $\Delta J=(J_{\mathrm{new}}-J_{\mathrm{old}})/|J_{\mathrm{old}}|\times100$ (negative = better).

## 5. Local sensitivity (`optimization/local_sensitivity.py`)

$$
\textbf{(S1)}\qquad S_i=\frac{J(\mathbf{x}_0+\delta_i\mathbf{e}_i)-J(\mathbf{x}_0-\delta_i\mathbf{e}_i)}{2\delta_i}
$$

with a one-sided difference (flagged `forward`/`backward`) when a perturbed point leaves the box or
violates a geometry constraint. Two normalisations are implemented:

- **(S2)** $\hat S_i=|S_i|/(\sum_j|S_j|+\epsilon)$ (`weighting="raw"`), and
- **(S2′, default)** $\hat S_i=|S_i|\delta_i/(\sum_j|S_j|\delta_j+\epsilon)$ (`weighting="step"`), i.e. the share of
  objective change produced by each parameter's own step. S2 compares derivatives across
  different units (mm, degrees, dimensionless) and is therefore not unit-invariant; S2′ is.

When the initial design is space-filling (LHS, the default) there are no one-at-a-time pairs, so the
HF sensitivities come from a ridge-regularised local linear fit over the evaluated designs near
$\mathbf{x}_0$ (`regression_sensitivity`):

$$
\textbf{(S1'')}\qquad \min_{a,\mathbf{s}}\ \sum_n \Big\|J(\mathbf{x}_n)-a-\sum_i s_i\,\frac{x_{n,i}-x_{0,i}}{\delta_i}\Big\|^2+\rho\|\mathbf{s}\|^2,
\qquad S_i=s_i/\delta_i ,
$$

followed by S2/S2′ as above (scheme flag `linear_fit`; the record stores the number of points and of
free parameters, because with about as many points as parameters the estimate is rough).

$J$ may be evaluated by HF data (`source="high_fidelity"`, S1 with the OAT initial design, S1'' with
LHS) or by the surrogate mean (`source="surrogate"`, recomputed at the incumbent every
iteration). The source is always recorded. These are **local response sensitivities**: they are not
causal physical effects unless backed by controlled perturbation experiments.

## 6. Counterfactual candidate generation (`optimization/counterfactual.py`)

Around the incumbent $\mathbf{c}$ (initially $\mathbf{x}_0$) with trust-region half-widths $\mathbf{r}$ (unit-cube units):

- **CF1 single-variable:** $\mathbf{c}\pm a\,\delta_i\mathbf{e}_i$, $a\in\mathcal{A}$ (default $\{0.5,1,2,3\}$), $i$ in the top-$k$ of $\hat S$.
- **CF2 pairwise:** $\mathbf{c}+a\delta_i\mathbf{e}_i+b\delta_j\mathbf{e}_j$ for all sign combinations, $(i,j)$ pairs of the top-$k$.
- **CF3 correlated:** $\mathbf{c}+t\,\mathbf{v}/\|\mathbf{v}\|\odot(\mathbf{x}^{\max}-\mathbf{x}^{\min})$ along schema-declared physical directions $\mathbf{v}$ and along the surrogate descent direction $\mathbf{v}=-\nabla_{\mathbf{u}}\mu_J(\mathbf{c})$.
- **CF4 latent:** steps along the first principal directions of the evaluated designs' $\mathbf{u}$ (a linear latent space); a learned encoder can supply directions via `latent_fn`.
- **CF5 exploration:** uniform samples in $\{\mathbf{u}:|\mathbf{u}-\mathbf{u}(\mathbf{c})|\le\mathbf{r}\}$.
- **CF6 surrogate-Pareto:** non-dominated points of NSGA-II run on the optimistic predicted objectives $\tilde\mu_k-\kappa\,\tilde\sigma_k$ inside the trust region.

Every candidate is projected onto the trust region and the box, snapped to the manufacturing grid,
de-duplicated against evaluated designs, and **rejected** if any geometry constraint is violated
(full polygon checks, Sec. 11). Each candidate records which parameters changed, by how much, and
the generator that proposed it.

## 7. Surrogate model (`models/`)

**Inputs (F1).** $\boldsymbol\phi(\mathbf{x})=[\mathbf{u}_{\mathcal F},\,\mathbf{r}(\mathbf{x}),\,\mathbf{e}(\mathbf{x}),\,\mathbf{a}(\mathbf{x}),\,\mathbf{p}(\mathbf{x})]$:
free unit-cube coordinates; ratios to the period ($L/P$, $W/P$); electrical sizes at the band-centre
reference frequency ($P/\lambda$, $h\sqrt{\varepsilon_r}/\lambda$, $L\sqrt{(\varepsilon_r+1)/2}/\lambda$); rotation as
$(\cos2\psi,\sin2\psi)$; optional prior-informed features (Sec. 7.1). The feature map deliberately
excludes closed-form resonance formulas so that results on the synthetic oracle are not inflated.

**Targets.** $\mathbf{t}=[\mathbf{J}\,|\,\mathbf{y}_{\mathrm{resp}}]$: the objective block plus a response block of
$(\mathrm{Re},\mathrm{Im})$ S-parameter spectra and specular RCS-reduction spectra resampled on $F$
frequencies in the band (`models/target_encoding.py`). Objectives are taken from the J-block; the
response block carries the physics losses, plots and multi-task regularisation.

**Members.** random forest (std across trees), XGBoost (bootstrap ensemble), Gaussian process
(Matérn-5/2 ARD + white noise; one GP per objective, one shared GP for the PCA-compressed response
block), MLP deep ensemble, physics-informed MLP deep ensemble, CNN on the rasterised cell mask
plus scalar features, transformer with one token per feature. Non-neural members regress the
response block in a PCA basis (default 8 components).

**(E1–E2) Ensemble.** For $M$ predictors (each network of a deep ensemble counts separately):

$$
\mu(\mathbf{x})=\frac1M\sum_m\hat f_m(\mathbf{x}),\qquad
\sigma^2(\mathbf{x})=\frac1M\sum_m(\hat f_m(\mathbf{x})-\mu(\mathbf{x}))^2 .
$$

Optionally (`include_member_variance`) the mean member predictive variance is added. The proposed
default ensemble is {physics-informed MLP (3 nets), GP, XGBoost}. When the structure is declared
passive, decoded complex reflections are projected onto $|S|\le1$ after averaging.

**Composite uncertainty.** $\sigma_J=\sqrt{\sum_k(\lambda_k s_k\sigma_k)^2}$ with $s_k=\partial\tilde J_k/\partial J_k$,
which assumes independent objective errors (stated approximation).

### 7.1 Transfer learning (`models/transfer.py`, `data/public_datasets/`)

Public data is treated as generic prior knowledge. A cell prior model
$g_\theta(L/P,W/P,Pf/c,hf/c,\varepsilon_r,\tan\delta,\cos2\psi,\sin2\psi)\to(\mathrm{Re}\,\Gamma,\mathrm{Im}\,\Gamma)_{p}$
is pre-trained on cell-level prior data (dimensionless, frequency-explicit inputs, so any frequency
grid is compatible), and optionally fine-tuned on the target's own cell spectra. For a target design
the cells listed in `schema.metadata.cells` are evaluated at $K$ band frequencies and appended to
$\boldsymbol\phi$ as $\mathbf{p}(\mathbf{x})=[\cos\Delta\varphi_{AB}(f_k),\sin\Delta\varphi_{AB}(f_k)]_k\cup\{\overline{|\Gamma_c|}\}$. Without prior data the block is absent.
The FSSDataBase adapter (`fssdatabase.py`) parses the published layout (per-sample mask and result
CSVs) for mask-based representation learning; it was written from the dataset card and could not be
verified against the files from the build environment.

## 8. Uncertainty estimation

Ensemble disagreement (E2) is the epistemic uncertainty used by acquisition. Its quality is
measured out of sample: every proposed design's prediction is stored *before* it is evaluated, and
the RMSE/MAE and $1\sigma/2\sigma$ coverage (`evaluation/metrics.calibration`) are reported.

## 9. Acquisition function (`optimization/acquisition.py`)

With $J^*$ the best HF composite so far, $\xi\ge0$, $z=(J^*-\mu-\xi)/\sigma$:

- **(A1)** $EI=(J^*-\mu-\xi)\Phi(z)+\sigma\varphi(z)$
- **(A2, proposed)** $A=EI+\beta\sigma+\gamma D$, $D(\mathbf{x})=\min_j\|\mathbf{u}_{\mathcal F}(\mathbf{x})-\mathbf{u}_{\mathcal F}(\mathbf{x}_j)\|_2/\sqrt{d_f}$ over evaluated and already-selected designs
- **(A3)** LCB: select $\arg\min\,\mu-\kappa\sigma$
- **(A4)** mean only: select $\arg\min\mu$ (ablation A5, no uncertainty)
- **(A5)** random selection (ablation A7)

Batches of $K$ are selected greedily, adding each pick to the reference set of $D$. In
multi-objective mode candidates are first ranked by Pareto rank of $\tilde\mu_k-\kappa\tilde\sigma_k$ and the
acquisition orders candidates within the best available rank.

*Defaults:* $\beta=0$, $\gamma=0.05$, $\kappa=1$. EI already contains an exploration term; the extra
$\beta\sigma$ term caused over-exploration with small data on the synthetic benchmark and is therefore
off by default but available.

## 10. Active-learning loop (Algorithm 1, `optimization/active_learning.py`)

```
Input: x0, y0, schema, objectives, physics config, N_max, K
 0  J0 <- J(y0); fit N3 normaliser; D <- {(x0, y0)}; c <- x0; r <- r0
 1  initial local design (charged): LHS of n_init points in the trust region (default)
    or one-at-a-time central perturbations x0 +/- delta_i e_i (gives HF S1)
    fit N1/N2 statistics if selected; c <- best evaluated
 2  repeat
 a    train the surrogate ensemble on D (physics loss L1 on labels and collocation designs)
 b    surrogate sensitivities S1/S2' and gradient at c
 c    pool C <- CF1..CF5(c, r, S) U CF6; reject invalid geometries
 d    predict mu, sigma (E1-E2) and composite (C1); audit predicted physics
 e    select K by acquisition A1-A5 (Pareto-ranked in multi-objective mode)
 f    [Mode B] evaluate with the HF source; log out-of-sample prediction errors
 g    D <- D U new; if best improved by > 1e-3|best|: c <- best, r <- min(1.5 r, r_max)
      else after 2 consecutive failures: r <- max(0.5 r, r_min)
 h    stop if any enabled rule T1-T4 fires
 3  return best validated design, validated Pareto set, ranked PREDICTED candidates
```

Defaults: $r_0=0.06$, $r_{\max}=0.3$, $r_{\min}=0.02$, $K=4$, $n_{\mathrm{init}}=8$.
**Mode A** (prediction-only) runs steps 2a–2e on existing data and returns ranked PREDICTED
candidates; it needs at least two evaluated designs. **Mode B** is either automatic (solver
adapter) or human-in-the-loop: `initial_design()` and `propose()` return designs to simulate,
`observe()` ingests uploaded results. The framework never treats an ML prediction as a
simulation: PREDICTED responses are rejected by `observe()` and by the dataset.

## 11. Physics constraints (`physics/`, `models/torch_common.py`)

Applied only when the stated assumptions hold (`PhysicsConfig.passive`, `.reciprocal`):

- **(P1) passivity:** $|S_{11}|^2+|S_{21}|^2\le1$ (reflection only: $|S_{11}|\le1$), power-normalised waves, same port modes; loss $L_{\mathrm{pass}}=\overline{\max(0,|S_{11}|^2+|S_{21}|^2-1)^2}$.
- **(P2) reciprocity:** $S_{21}=S_{12}$; loss $\overline{|S_{21}-S_{12}|^2}$; only for reciprocal media with matching port definitions.
- **(P3) phase continuity (optional):** magnitude-weighted change of the unit phasor between adjacent samples.
- **(P4) smoothness:** $L_{\mathrm{smooth}}=\overline{w_f\,h(\hat y_{f+1}-2\hat y_f+\hat y_{f-1})}$, with $h$ the square or a Huber function (default) and resonance-preserving weights $w_f=1/(1+(c_f/c_{\mathrm{ref}})^2)$, $c_f$ the curvature of the training spectra and $c_{\mathrm{ref}}$ its 75th percentile; RCS spectra in dB are scaled by 1/10.
- **(G1) geometry:** $J_{\mathrm{constraint}}=\sum_i\max(0,g_i(\mathbf{x}))^2$ with analytic template constraints (gaps to periodic neighbours accounting for rotation, minimum width, slot margins) and schema linear constraints; generic polygon checks (self-intersection, overlap, periodic minimum gap, minimum feature via morphological opening, disconnected metal, footprint, substrate thickness) reject designs before ranking and before export.

**(L1) Training loss of the physics-informed networks:**

$$
L=L_{\mathrm{data}}+\lambda_p(L_{\mathrm{pass}}+L_{\mathrm{rec}})+\lambda_sL_{\mathrm{smooth}}+\lambda_{\varphi}L_{\mathrm{phase}},
\qquad L_{\mathrm{data}}=\frac{\sum_{n,j}\omega_j m_{nj}(\hat t_{nj}-t_{nj})^2}{\sum_{n,j}\omega_jm_{nj}}
$$

on standardised targets ($m_{nj}$ masks missing entries; $\omega_j=3$ on the J-block, 1 elsewhere).
Physics terms use de-standardised outputs and are averaged over the training designs and $N_c$
unlabeled collocation designs sampled in the trust region (no labels needed). The geometry term
$\lambda_gL_{\mathrm{geometry}}$ of the general loss acts in the optimisation objective (C1, G1) rather than in
training, because every training design is a valid geometry by construction. Each term has its own
switch for ablation, and the full configuration is stored with each experiment.

## 12. Multi-objective optimisation (`optimization/nsga2.py`, `pareto.py`)

NSGA-II (fast non-dominated sorting, crowding distance, SBX $\eta_c=15$, polynomial mutation
$\eta_m=20$, constraint domination) is used (i) on the surrogate inside the trust region (CF6), (ii)
as an HF black-box baseline. The validated Pareto set over all evaluated designs is reported with
the user's choice of final candidate left open. Hypervolume (exact recursive slicing) is computed in
normalised objective space with reference point $\tilde{\mathbf{J}}(\mathbf{x}_0)+h$ ($h=0.1$), i.e. the
volume of improvements that do not fall more than $h$ behind the baseline on any objective.

## 13. Stopping criterion (`optimization/stopping.py`)

Any enabled rule stops the loop; the triggering rule is recorded:
**(T1)** $|J^*_t-J^*_{t-k}|<\epsilon_J$; **(T2)** $\Delta HV<\epsilon_{HV}$ for $p$ consecutive iterations;
**(T3)** $N_{\mathrm{HF}}\ge N_{\max}$; **(T4)** max $\sigma$ of the selected batch $<\sigma_{\min}$.

## 14. Complexity

Per iteration with $n$ evaluated designs, pool size $|C|$ and $M$ ensemble predictors:
GP training $O(n^3)$ per objective; network training $O(E\,(n+N_c)\,P)$ for $E$ epochs and $P$
parameters; XGBoost/RF $O(T\,n\log n)$; prediction $O(|C|M)$; NSGA-II on the surrogate
$O(G\,N_{\mathrm{pop}}M + G\,N_{\mathrm{pop}}^2m)$; geometry validation $O(|C|)$ polygon operations. HF
cost: $n_{\mathrm{init}}+K\cdot T$ evaluations for $T$ iterations, which dominates in practice.

## 15. Evaluation protocol and metrics (`evaluation/`)

Prediction: MAE, RMSE, $R^2$, coverage. Optimisation: $\Delta$RCS, $\Delta$BW (O7), $\Delta H_{\mathrm{ang}}$, $\Delta J$.
Efficiency: $E=\Delta J/N_{\mathrm{HF}}$ (improvement per HF simulation; $\Delta J>0$ is improvement), and
the HF evaluations needed to reach the proposed method's median final value. Every algorithm gets
the same baseline response and the same $N_{\max}$; invalid geometries are rejected without
consuming budget. Results over all seeds are reported (mean, std, median, best, worst, bootstrap
95 % CI); no run is discarded. Hyper-parameter defaults were chosen on tuning seeds that are
disjoint from the seeds used for the reported statistics.

External evaluation (e.g. a design recovered from a paper) uses the partitions TRAINING /
VALIDATION / EXTERNAL_EVALUATION / REFERENCE_ONLY (`data/dataset.py`): the optimiser sees only
the baseline design and response; a paper's optimised design is stored as REFERENCE_ONLY and the
dataset refuses to train on it.

## 16. Limitations

- The synthetic oracle (`solvers/analytic_oracle.py`) is a quasi-static high-impedance-surface
  model with scalar physical optics: no mutual coupling between dissimilar cells, edge
  diffraction, surface waves or cross-polarisation. It demonstrates the method; it is not evidence
  of physical performance. The synthetic cell prior shares its physics family, so transfer gains
  measured with it are optimistic.
- Surrogates trained on tens of points are weak outside the sampled region; the trust region and
  the out-of-sample error log are the safeguards, not guarantees.
- Composite uncertainty assumes independent objective errors.
- Local sensitivities depend on the step sizes and on the incumbent; they are not global or causal.
- Geometry inference from images/STL is approximate and non-unique; parameters must be confirmed.
- Plot-digitised responses are approximate, with resolution-limited precision.
- HFSS/CST automation is not implemented; full-wave validation is via upload (Mode B).
- Report limitations are detected automatically (`evaluation/report_generator.detect_limitations`).
