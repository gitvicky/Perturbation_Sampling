# PAPER.md — Instructions for Writing the Paper

This file provides Claude with the structure, conventions, and automation instructions for producing the ICLR 2026 paper in LaTeX. The template is `iclr2026_conference.tex` with style files already in `Paper/`.

---

## 1. LaTeX Conventions

- Use the existing `iclr2026_conference.sty` style — do not modify formatting parameters.
- Math macros are defined in `math_commands.tex` (from Goodfellow's dlbook notation). Use `\va`, `\mA`, `\vx`, `\vtheta`, `\E`, `\KL`, `\R`, etc. where applicable.
- Citations use `natbib`: `\citet{}` for in-text, `\citep{}` for parenthetical. Bib entries go in `iclr2026_conference.bib`.
- Figures: use `\includegraphics[width=0.X\linewidth]{figures/filename.pdf}`. All experiment figures should be saved as PDF to `Paper/images/`.
- Tables: use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). Numbers should be formatted to consistent decimal places, best results in **bold**.
- The main text must fit within **9 pages** (10 for camera-ready). Appendix is unlimited.

---

## 2. Paper Structure

The paper should be organized into the following sections within `iclr2026_conference.tex`:

### Front Matter
- **Title**: Working title — "Inverting Physics Residual Bounds: Perturbation Sampling for Calibrated Uncertainty in Neural PDE Solvers" (refine as needed).
- **Abstract**: ~150 words. State the problem (CP gives residual-space guarantees, not physical-space), the method (perturbation sampling + advanced gradient-guided variants), key results (coverage, scalability), and significance.

### Section 1: Introduction
- Motivation: Neural PDE/ODE surrogates lack calibrated UQ.
- Gap: PRE/CP from Gopakumar et al. (2025) calibrates in residual space — but users need physical-space bounds.
- Contribution summary (3-4 bullet points): (i) inversion framework, (ii) scalable sampling methods, (iii) empirical validation across linear/nonlinear ODEs and PDEs.

### Section 2: Background
- **2.1 Neural PDE/ODE Surrogates**: Problem setup, notation.
- **2.2 Physics Residual Error (PRE)**: FD-stencil convolution operators, `ConvOperator` formulation.
- **2.3 Conformal Prediction**: Split CP, nonconformity scores, `qhat` calibration. Reference `2502.04406v2`.

### Section 3: Method — Perturbation Sampling for Bound Inversion
- **3.1 Problem Formulation**: Given calibrated residual bounds `[-qhat, +qhat]`, find physical-space bounds `[u_lo, u_hi]` such that coverage is preserved.
- **3.2 Standard Rejection Sampling**: Monte Carlo perturbation + binary accept/reject. State the containment check.
- **3.3 Differentiable Rejection (Optimization)**: Gradient rescue of rejected samples. Loss function, backprop through the physics operator.
- **3.4 Posterior Sampling (Langevin Dynamics)**: Langevin SDE formulation, step size, residual-guided gradient.
- **3.5 Generative Modeling**: MLP/CNN trained to map N(0,I) to valid manifold. Training objective.

Each method subsection should include:
- A mathematical description (equations in `align` or `equation` environments).
- An algorithm box (`\begin{algorithm}...\end{algorithm}` using `algorithmic` package — add `\usepackage{algorithm,algorithmic}` to the preamble).
- Complexity/cost discussion (one sentence).

### Section 4: Experiments
- **4.1 Experimental Setup**: Describe the test problems, training details, noise types, and evaluation protocol.
- **4.2 ODE Experiments** (SHO, DHO, Duffing): Coverage curves, bounds plots, acceptance rates.
- **4.3 PDE Experiments** (1D Advection): Scaling to spatiotemporal fields.
- **4.4 Results Summary Table**: Single table comparing all methods across all problems.

### Section 5: Ablation Studies
- **5.1 Noise Type**: Effect of spatial vs. white vs. GP vs. B-spline noise on coverage and band width.
- **5.2 Number of Samples**: Convergence of coverage and bound width vs. `n_samples`.
- **5.3 Optimization Hyperparameters**: Learning rate, number of rescue steps, Langevin step size.
- **5.4 Coverage Level**: Vary `alpha` and show coverage tracks target.

### Section 6: Discussion & Related Work
- Comparison with ensemble methods, MC dropout, evidential deep learning.
- Limitations: computational cost, assumptions on operator differentiability.

### Section 7: Conclusion

### Appendix
- Extended derivations, additional plots, full hyperparameter tables.

---

## 3. Experiment Automation

When asked to generate results for the paper, Claude should run experiments and capture outputs as follows:

### 3.1 Running Experiments

```bash
# Activate env
source .venv/bin/activate

# ODE experiments — all methods
python Expts/experiment_runner.py sho --noise-type spatial
python Expts/experiment_runner.py sho --use-optimization --noise-type spatial
python Expts/experiment_runner.py sho --use-langevin --noise-type spatial
python Expts/experiment_runner.py sho --use-generator --noise-type spatial

python Expts/experiment_runner.py dho --noise-type spatial
python Expts/experiment_runner.py dho --use-optimization --noise-type spatial
python Expts/experiment_runner.py dho --use-langevin --noise-type spatial
python Expts/experiment_runner.py dho --use-generator --noise-type spatial

python Expts/experiment_runner.py duffing --noise-type spatial
python Expts/experiment_runner.py duffing --use-optimization --noise-type spatial
python Expts/experiment_runner.py duffing --use-langevin --noise-type spatial
python Expts/experiment_runner.py duffing --use-generator --noise-type spatial

# PDE experiment
python Expts/Advection_Perturb.py
```

### 3.2 Extracting Results

After running each experiment, extract from the console output and/or saved files:

| Metric | Description | Where to put it |
|--------|-------------|-----------------|
| **Empirical coverage** | Fraction of truth within bounds at target alpha | Results table (Section 4.4) |
| **Acceptance rate** | Fraction of samples accepted (rejection methods) | Results table |
| **Mean band width** | Average `u_hi - u_lo` across time/space | Results table |
| **Wall-clock time** | Time for perturbation sampling step | Results table |
| **Coverage curve** | Coverage vs. alpha plot | Figure (Section 4.2/4.3) |
| **Bounds plot** | Prediction + bounds + truth trajectory | Figure (Section 4.2/4.3) |

### 3.3 Generating Figures

- Save all figures to `Paper/images/` as PDF (use `plt.savefig('Paper/images/name.pdf')`).
- Naming convention: `{problem}_{method}_{plot_type}.pdf`, e.g., `sho_mc_bounds.pdf`, `duffing_langevin_coverage.pdf`.
- For comparison figures, use consistent color palette from `experiment_runner.py` (`PALETTE` dict).

### 3.4 Generating Tables

Format the results summary table in LaTeX as:

```latex
\begin{table}[t]
\caption{Empirical coverage and efficiency across all test problems at $1-\alpha = 0.9$.}
\label{tab:main-results}
\begin{center}
\begin{tabular}{llcccc}
\toprule
\textbf{Problem} & \textbf{Method} & \textbf{Coverage} & \textbf{Accept. Rate} & \textbf{Band Width} & \textbf{Time (s)} \\
\midrule
SHO & MC & & & & \\
    & Optim & & & & \\
    & Langevin & & & & \\
    & Generative & & & & \\
\midrule
DHO & MC & & & & \\
% ... etc.
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

### 3.5 Ablation Tables and Figures

For each ablation (Section 5), produce:
- One table or one figure (whichever communicates the result more clearly).
- Ablations over noise type: run all 4 noise types (`spatial`, `white`, `gp`, `bspline`) for a single problem (SHO recommended) and single method (MC recommended).
- Ablations over n_samples: modify `PerturbationSamplingConfig.n_samples` in a sweep `[1000, 5000, 10000, 20000, 50000]`.
- Ablations over alpha: sweep `alpha` in `[0.01, 0.05, 0.1, 0.2, 0.3]`.

---

## 4. Writing Workflow

When asked to write or update a section:

1. **Read** the current state of `iclr2026_conference.tex` to see what exists.
2. **Run** the relevant experiments if results are needed and not yet available.
3. **Write** the LaTeX content using the structure above, inserting actual numbers from experiments.
4. **Edit** `iclr2026_conference.tex` to replace/add the section content.
5. **Compile** with `pdflatex iclr2026_conference.tex && bibtex iclr2026_conference && pdflatex iclr2026_conference.tex && pdflatex iclr2026_conference.tex` to verify it builds.

### Writing Style Guidelines
- Formal academic tone, third person ("we propose", "the method achieves").
- Present tense for methods ("we sample perturbations"), past tense for experiments ("coverage reached 92%").
- Keep sentences concise. Avoid filler phrases ("It is worth noting that...").
- Every claim must be backed by either a citation, a theorem, or experimental evidence from our runs.
- Define notation on first use and keep it consistent with `math_commands.tex`.

---

## 5. BibTeX References

Key references to include in `iclr2026_conference.bib`:

- **Gopakumar et al. (2025)**: "Calibrated Physics-Informed Uncertainty Quantification" — the PRE/CP paper we extend. `2502.04406v2`.
- **Vovk et al. (2005)**: Algorithmic Learning in a Random World — foundational CP reference.
- **Romano et al. (2019)**: Conformal quantile regression.
- **Raissi et al. (2019)**: Physics-informed neural networks.
- **Chen et al. (2018)**: Neural ordinary differential equations.
- **Li et al. (2021)**: Fourier neural operator.
- **Welling & Teh (2011)**: Bayesian learning via stochastic gradient Langevin dynamics.

Add entries as needed when writing each section. Keep the `.bib` file alphabetically ordered.

---

## 6. Figure Checklist

Before finalizing any figure for the paper:
- [ ] Font size readable at column width
- [ ] Axis labels include units where applicable
- [ ] Legend does not obscure data
- [ ] Colorblind-friendly palette (the project PALETTE is suitable)
- [ ] Saved as vector PDF, not raster PNG
- [ ] Caption is self-contained (reader can understand without reading main text)
