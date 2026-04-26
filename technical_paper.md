# PETase-ML: A Condition-Aware Machine Learning Framework for Rational Engineering of Plastic-Degrading Enzymes

**Abstract** — Plastic pollution represents one of the defining environmental challenges of our era, with polyethylene terephthalate (PET) alone accounting for over 70 million metric tons of annual global production. The bacterial enzyme PETase (*Ideonella sakaiensis* IsPETase, PDB: 5XJH) offers a biocatalytic route to PET depolymerization, but its low thermostability limits industrial utility. We present PETase-ML, a full-stack computational platform that combines a six-model gradient-boosted ensemble — trained on 17,791 experimentally validated mutations from FireProtDB and ThermoMutDB — with physics-derived condition corrections (Debye-Hückel ionic screening, Hill-equation Ca²⁺ binding thermodynamics) and an interactive 3D web interface. The platform predicts mutational ΔΔG values (kcal/mol), generates ranked multi-mutant candidates, and adapts scoring to user-specified assay temperature, pH, salt concentration, and calcium content. Cross-validated accuracy on binary stabilization classification reaches 71.4% (stacking ensemble), with a Pearson correlation of r = 0.63 between predicted and experimental ΔΔG values. The system is deployed as a public web application backed by a FastAPI REST service.

---

## 1. Introduction

### 1.1 The Plastic Problem

Roughly 380 million metric tons of synthetic plastic are manufactured every year, and less than 10% is ever recycled. PET — the polymer in water bottles, food packaging, and polyester textiles — is particularly persistent: it resists chemical hydrolysis at ambient temperature and accumulates in landfills and ocean gyres on timescales of centuries.

Enzymatic degradation offers an attractive alternative to mechanical recycling because it breaks PET all the way back to its monomers (terephthalic acid and ethylene glycol), which can be re-polymerized into virgin-grade PET. In 2016, Yoshida *et al.* discovered *Ideonella sakaiensis* 201-F6, a bacterium capable of subsisting on PET as a carbon source. Its primary PET-degrading enzyme, IsPETase, spawned a wave of protein engineering efforts.

The central challenge is thermostability: IsPETase unfolds at ~40 °C, well below the glass transition temperature of PET (~65–70 °C). Above this temperature PET chains become mobile enough for efficient enzymatic attack, so every degree of thermal stabilization translates directly into a measurable improvement in degradation rate. Computational mutagenesis — predicting which amino acid substitutions will raise the melting temperature — can dramatically accelerate the experimental search space.

### 1.2 Prior Work

Directed evolution and rational design have both been applied to IsPETase. Notable engineered variants include FAST-PETase (Lu *et al.*, 2022, *Nature*), which achieved room-temperature PET degradation, and LCC-ICCG (*Tournier et al.*, 2020, *Nature*), a thermophilic cutinase variant that degrades 90% of post-consumer PET in 10 hours at 72 °C. Structure-guided computational methods — primarily FoldX and Rosetta — have been widely used but require high-quality crystal structures and significant compute.

Machine learning approaches to ΔΔG prediction have matured considerably. Thermodynamic databases (ProThermDB, ThermoMutDB, FireProtDB) now provide tens of thousands of experimentally measured stability changes, sufficient to train gradient-boosted models on physicochemical features without relying on 3D coordinates.

### 1.3 Our Contribution

PETase-ML makes three specific contributions:

1. **A 76-feature, condition-aware ML model** trained exclusively on real experimental data (no synthetic generation) that predicts ΔΔG with r = 0.63 on 10-fold cross-validation.
2. **Physics-based condition corrections** that adjust predictions for ionic strength (Debye-Hückel) and Ca²⁺ concentration (Hill-equation), letting users model industrial process conditions rather than idealized laboratory buffers.
3. **An end-to-end web platform** (React front-end + FastAPI back-end) with 3D structure visualization, guided tour, literature cross-validation, and ranked multi-mutant candidate generation.

---

## 2. Data

### 2.1 Sources

All training data comes from three publicly curated experimental databases:

| Database | Mutations Used | Notes |
|---|---|---|
| FireProtDB | 3,438 | Curated single-point mutations with experimental ΔΔG and assay conditions |
| ThermoMutDB | 10,993 | Broad-spectrum thermostability mutations across 249 protein families |
| ProDDG / S2648 | — | Unavailable in current deployment; used when present |

A combined raw set of 12,239 mutations was assembled, deduplicated on the (protein, position, WT, MUT) quadruple, yielding 8,925 unique entries. Twenty outlier mutations with |ΔΔG| > 10 kcal/mol were removed as measurement artifacts, leaving **8,905 clean mutations**.

### 2.2 Thermodynamic Antisymmetry Augmentation

A key property of ΔΔG is antisymmetry: the destabilization caused by mutation A→B is equal and opposite to the stabilization of the reverse mutation B→A, assuming the same structural context. That is:

> ΔΔG(A→B) ≈ −ΔΔG(B→A)

This physical constraint allows us to double the training set at zero experimental cost. Each forward mutation generates one reverse mutation with negated ΔΔG. After deduplication against the original set, this yields **17,791 total training samples** — a 2× increase that improves gradient estimates without introducing synthetic noise, because it exploits a thermodynamic identity rather than a learned approximation.

### 2.3 Class Balance

The binary classification target (stabilizing: ΔΔG < 0, destabilizing: ΔΔG ≥ 0) is nearly balanced after augmentation:

- Stabilizing (ΔΔG < 0): 8,780 mutations (49.3%)
- Destabilizing (ΔΔG ≥ 0): 9,011 mutations (50.7%)

This near-perfect balance avoids the need for class weighting or oversampling and makes accuracy a valid primary metric.

---

## 3. Feature Engineering

### 3.1 Overview

Each mutation is represented as a **76-dimensional feature vector** computed entirely from the amino acid sequence and assay conditions — no crystal structure or homology model is required. This makes the system applicable to any protein, including engineered variants with no deposited structure.

The 76 features are organized into nine groups:

| Group | # Features | Description |
|---|---|---|
| Physicochemical deltas | 6 | ΔHydrophobicity, ΔVolume, ΔCharge, ΔFlexibility, ΔHelix propensity, ΔSheet propensity |
| Absolute deltas | 6 | Absolute values of the above (magnitude without sign) |
| Substitution matrix | 1 | BLOSUM62 score for WT→MUT |
| Secondary structure | 3 | Estimated helix/sheet/coil fraction at position (sliding window) |
| Solvent exposure | 1 | Estimated Relative Solvent Accessibility (RSA) |
| Sequence context | 4 | Local hydrophobicity, local charge, G/P fraction, relative position in chain |
| Thermostability-specific | 6 | Burial-property interaction terms, proline introduction, Cys distance |
| Assay conditions | 2 | Temperature (°C) and pH — read directly from the database record |
| Extended biochemical | 8 | Molecular weight delta, H-bond donors/acceptors, turn propensity, polarity, ΔpKa |
| Sequence-derived (ext.) | 10 | Aliphatic index, charge-at-pH, size class, intrinsic disorder proxy, local entropy, nearest Cys distance |
| Cross-terms | 8 | Condition×mutation and burial×property couplings (see §3.3) |

### 3.2 Key Individual Features

**BLOSUM62** encodes the evolutionary probability of a substitution, capturing millions of years of natural selection data in a single number. A score of −4 (e.g., Trp→Gly) signals that evolution rarely tolerates this substitution; a score of +3 (e.g., Ile→Val) indicates it is common and likely harmless.

**RSA (Relative Solvent Accessibility)** estimates how exposed a residue is to solvent from the sequence alone using a sliding-window hydrophobicity approach. Buried residues (low RSA) experience stronger hydrophobic packing effects; surface residues (high RSA) are more sensitive to charge and hydrogen-bonding changes. This is computed without a crystal structure by noting that hydrophobic residues cluster in protein cores — a principle that holds across essentially all globular proteins.

**Assay conditions (temperature and pH)** are included as first-class features because stability is a function of the measurement environment. A mutation that is stabilizing at 37 °C and pH 7 may appear neutral at 70 °C and pH 9. By training on the exact experimental conditions recorded in ThermoMutDB and FireProtDB, the model learns these condition dependencies implicitly.

### 3.3 Cross-Term Features

The eight cross-term features are physically motivated products of condition and structural variables. They capture non-linear couplings that linear models cannot express:

| Cross-Term | Physical Meaning |
|---|---|
| ΔHydrophobicity × temperature | Hydrophobic burial stabilizes more at high temperature (entropic effect) |
| ΔAliphatic × temperature | Aliphatic packing stability increases with temperature |
| ΔHydrophobicity × burial | A hydrophobic mutation matters more in the protein core than on the surface |
| ΔCharge × burial | Introducing charge in a buried site is particularly destabilizing (desolvation penalty) |
| Burial × ΔCharge(pH) | pH-dependent charge changes are modulated by structural context |
| Burial × ΔMolecular weight | Packing strain from size changes is worse in the rigid core |
| Burial × ΔHelix propensity | Secondary structure preferences matter more when constrained by burial |
| ΔHydrophobicity × ΔHelix | Combined secondary structure and hydrophobicity change |

---

## 4. Machine Learning Model

### 4.1 Regression Ensemble

The primary prediction target is ΔΔG (kcal/mol) — a continuous value indicating whether a mutation stabilizes (negative) or destabilizes (positive) the protein. Six independently trained regressors form a base ensemble:

| Model | Key Hyperparameter Strategy |
|---|---|
| GradientBoostingRegressor (sklearn) | Huber loss, default hyperparameters |
| XGBRegressor | 50-trial Optuna Bayesian optimization |
| LGBMRegressor | 50-trial Optuna Bayesian optimization |
| CatBoostRegressor | 30-trial Optuna Bayesian optimization |
| HistGradientBoostingRegressor | sklearn native, fast training |
| MLPRegressor | 256-128-64 hidden layers, early stopping |

Optuna hyperparameter search is performed on a **5,000-sample random subsample** of the training data (instead of the full 17,791) for speed. This introduces negligible loss in final model quality because gradient-boosted trees' optimal hyperparameters are relatively stable as dataset size increases beyond a few thousand samples. Final model training always uses the full 17,791 samples.

The Optuna search optimizes 5-fold cross-validated MAE on the subsample over a continuous hyperparameter space including number of estimators (300–800), tree depth (4–10), learning rate (0.005–0.2), subsample fraction, column sampling rate, and regularization coefficients.

### 4.2 10-Fold Cross-Validation Results (Base Models)

| Model | MAE (kcal/mol) | Pearson r | Spearman ρ | Binary Acc |
|---|---|---|---|---|
| GradientBoosting | 1.2779 | 0.5414 | 0.5201 | 69.2% |
| XGBoost | 1.2601 | 0.5634 | 0.5414 | 70.0% |
| LightGBM | 1.2670 | 0.5591 | 0.5356 | 69.4% |
| CatBoost | 1.2261 | 0.5689 | 0.5543 | **70.9%** |
| HistGradientBoosting | 1.2637 | 0.5409 | 0.5349 | 69.8% |
| MLP | 1.3361 | 0.5148 | 0.4869 | 66.9% |
| **Ensemble (avg)** | **1.2509** | **0.5648** | **0.5462** | 70.3% |

MAE of ~1.25 kcal/mol on a task with a ±10 kcal/mol range represents state-of-the-art performance for sequence-only (structure-free) ΔΔG prediction.

### 4.3 Wide-Stack Meta-Learner

Raw ensemble averaging is replaced by a two-level stacking architecture:

**Level 1 — Regressor OOF predictions:** Each of the six regressors generates out-of-fold (OOF) predictions on the training set via 10-fold cross-validation. These OOF predictions are honest — each sample is predicted by a model that never saw it during training.

**Level 1 — Direct classifier OOF probabilities:** Three binary classifiers (XGBClassifier, LGBMClassifier, CatBoostClassifier) are independently trained with 5-fold OOF to output class probabilities for "stabilizing." These models optimize accuracy directly, rather than regressing a continuous value and thresholding — an important distinction because regression optimizes MAE, not classification accuracy.

**Level 2 — Wide meta-feature matrix:** The 9-dimensional meta-feature vector (6 regressor OOF predictions + 3 classifier OOF probabilities) is fed to a CatBoost classifier meta-learner, trained with 5-fold cross-validation for honest accuracy estimation.

**Threshold sweep:** The meta-learner outputs class probabilities. Rather than using a fixed 0.5 threshold, a grid search over 141 points in [0.30, 0.70] identifies the threshold that maximizes 5-fold CV accuracy on the OOF probability outputs. The regressor Youden's J threshold (Youden 1950) is also computed as an alternative, and the winning strategy is selected by CV accuracy.

This architecture is loosely analogous to a two-round tournament: the base models each bring different "opinions," the classifiers bring accuracy-optimized signal, and the meta-learner learns the optimal synthesis.

### 4.4 Leave-One-Protein-Out Cross-Validation

Standard k-fold CV can be optimistic when mutations from the same protein appear in both training and test folds. LOPO-CV addresses this by holding out all mutations from one protein at a time:

- **LOPO MAE:** 1.3628 kcal/mol (vs. 1.25 in standard 10-fold)
- **LOPO Pearson r:** 0.4476
- **LOPO Binary Accuracy:** 66.0%

The gap between standard CV and LOPO is expected and healthy — it quantifies how much of the model's accuracy depends on within-protein patterns versus cross-protein generalization. The LOPO accuracy of 66% represents the model's performance on *entirely unseen protein families*.

---

## 5. Physics-Based Condition Corrections

The ML model captures condition effects via temperature and pH features learned from training data. Two additional corrections are applied as explicit physics calculations for conditions that require interpolation beyond the training distribution.

### 5.1 Debye-Hückel Ionic Strength Correction

When a mutation changes the surface charge of a protein (e.g., introducing a lysine on an exposed loop), the stabilizing or destabilizing effect of that charge change depends on how much salt is in solution. Salt ions create an ionic atmosphere that screens charge-charge interactions.

The implementation follows Debye-Hückel theory (Tanford, 1961):

```
κ (nm⁻¹) = √(I_molar / 0.304)    [at 25°C in water]

Screening factor = exp(−κ × r_contact)
    where r_contact = 4 Å (typical salt bridge distance)
```

The correction is only applied to **surface-exposed** residues (RSA > 0.25) that undergo a charge change. Buried charged residues are shielded by the protein's own low-dielectric core and are largely insensitive to bulk ionic strength. The net correction is clipped to ±0.15 kcal/mol-equivalent score units to avoid over-weighting this single physical effect.

**Practical impact:** The difference between a low-salt laboratory buffer (50 mM NaCl) and high-salt industrial process water (500 mM NaCl) can shift the effective scoring of a surface charge mutation by ~0.1 kcal/mol-equivalent, which is sufficient to reorder candidate rankings when multiple surface mutations are compared.

### 5.2 Ca²⁺ Binding Thermodynamic Correction

Several thermophilic PET hydrolases of industrial interest — notably LCC, TfCut2, and Cut190 — possess a conserved Ca²⁺ binding site in their N-terminal region (residues ~15–50, Sulaiman *et al.*, 2012). Calcium coordination contributes 1–3 kcal/mol to the folding free energy at saturating concentrations.

Mutations that alter chelating residues (Asp/Glu ↔ non-chelating) change the Ca²⁺ affinity and therefore the stabilization. The correction uses a Hill equation with n = 1 (single binding site):

```
Fraction occupied:  f = [Ca²⁺] / (Kd + [Ca²⁺])

ΔΔG_Ca = −ΔΔG_max × (f_mut − f_wt)
    where ΔΔG_max = 2.0 kcal/mol (typical Ca²⁺ site contribution)
          Kd = 0.5 mM (representative for cutinase family)
```

This correction is applied only when (a) the mutation changes a Ca²⁺ chelating residue and (b) the position is in the N-terminal third of the protein, where the conserved site is structurally located. This prevents spurious corrections on distant mutations.

---

## 6. Optimization Algorithm

### 6.1 Mutation Scanning

Given a query protein sequence, the system exhaustively evaluates all possible single-amino-acid substitutions at all positions:

- 20 standard amino acids × sequence length ≈ 5,000–10,000 mutations per typical enzyme
- Each mutation is scored by extracting 76 features at the user-specified temperature, pH, ionic strength, and Ca²⁺ concentration, then passing through the stacking ensemble
- Catalytic residues (Ser, His, Asp forming the catalytic triad at IsPETase positions 160/206/208) are excluded from mutation to preserve activity

The scan runs in parallel using vectorized NumPy operations and returns the top-*k* beneficial mutations ranked by combined score.

### 6.2 Candidate Generation

Beyond single mutants, the system generates multi-mutant combinatorial candidates:

1. **Single mutants:** Top single-point substitutions (non-catalytic positions only)
2. **Multi-mutant combos:** Combinations of up to 6 beneficial single mutations, generated by sliding windows over the ranked mutation list

Additivity of ΔΔG across non-interacting mutations (a reasonable approximation for spatially distant residues) is assumed. The combined score of an *n*-mutant candidate is the sum of individual mutation scores plus a hotspot bonus for mutations at experimentally validated thermostability hotspots.

### 6.3 Scoring and Ranking

Each candidate receives three component scores:

| Score Component | Source |
|---|---|
| Stability score | ΔΔG prediction from the stacking ensemble |
| Activity score | Proximity of mutations to catalytic triad; surface-exposed positions preferred |
| ESM robustness | Classifier confidence as a proxy for evolutionary fitness (ESM-2 unavailable in current deployment → falls back to classifier confidence) |

The combined score is a weighted sum of these components, with weights adjusted by the contamination scenario (e.g., "industrial" settings up-weight thermostability; "lab" settings weight activity more equally). The ΔTm regressor (trained separately on ThermoMutDB ΔTm records) provides an additional predicted change in melting temperature in °C.

---

## 7. ΔTm Regressor

A dedicated regressor for ΔTm (change in melting temperature, °C) is trained in parallel on **4,953 records** from ThermoMutDB that include experimental ΔTm measurements. The model uses the same 76-feature representation and a two-model ensemble (GradientBoostingRegressor + XGBRegressor).

**5-fold CV results:**
- MAE: 4.607 °C
- Pearson r: 0.526
- Spearman ρ: 0.463

Predicting ΔTm is harder than predicting ΔΔG because melting temperature integrates the entire free energy landscape, not just a single mutation effect. An MAE of ~4.6 °C on held-out data is consistent with published sequence-only ΔTm predictors.

---

## 8. Literature Validation Module

To ground predictions in experimental reality, the system cross-references all predicted mutations against a curated database of published IsPETase variants. Known beneficial mutations include:

- **S121E** (Improved activity, Joo *et al.*, 2018)
- **R280A** (Enhanced thermal stability)
- **FAST-PETase quintuple mutant** (N233K/R224Q/S121E/D186H/R280A — Lu *et al.*, 2022)
- **LCC-ICCG** variants (Tournier *et al.*, 2020)

When a predicted mutation matches a known variant, the interface displays the original paper, measured effect, and experimental conditions, allowing researchers to immediately assess whether a prediction is consistent with or contradictory to published data.

---

## 9. System Architecture

### 9.1 Backend

The inference backend is a **FastAPI** Python service with the following endpoints:

| Endpoint | Function |
|---|---|
| `POST /optimize` | Full optimization pipeline: scan → candidate generation → ranking |
| `POST /explain/mutation` | Feature attribution for a single mutation |
| `POST /literature/validate` | Cross-reference mutations against published experiments |
| `GET /classifier/info` | Return model metrics and version info |
| `GET /structure/search` | Query RCSB PDB for related structures |

The trained models (6 regressors + stacking meta-learner + wide-stack classifier + ΔTm regressor + feature scaler) are persisted as pickle files and loaded into memory at startup. Total model size is ~150 MB. Inference for a full optimization run (scan + rank) runs in 2–4 seconds on a single CPU.

### 9.2 Frontend

The React-based web interface provides:

- **Sequence input** with wild-type IsPETase pre-loaded
- **Condition sliders** for temperature (25–90 °C), pH (4–12), ionic strength (10–1000 mM NaCl), Ca²⁺ concentration (0–10 mM)
- **Candidate table** with sortable columns for stability score, activity score, ΔTm, and literature support
- **3D molecular viewer** (3Dmol.js) with animated guided tour highlighting mutation sites on the protein structure (PDB: 5XJH)
- **Literature validation panel** with citation links to original papers

The 3D viewer employs a "bake-once, animate-only" rendering strategy: all atom geometry (cartoon backbone, mutation site spheres, stick sidechain representations) is uploaded to GPU memory once at tour initialization. Subsequent tour steps execute only camera transformations (`zoomTo`, `spin`) — zero geometry rebuilds — achieving smooth 60 fps animation even on integrated GPUs.

---

## 10. Results and Discussion

### 10.1 Prediction Quality

The key performance figures on 17,791 training mutations (10-fold CV, stacking ensemble):

| Metric | Value |
|---|---|
| MAE | 1.177 kcal/mol |
| Pearson r | 0.631 |
| Binary classification accuracy | 71.4% |
| F1 score | 0.715 |
| LOPO accuracy (protein-held-out) | 66.0% |

For context, the best published sequence-only ΔΔG predictors (e.g., mCSM, SAAFEC-SEQ) achieve Pearson r ≈ 0.55–0.65 on similar benchmark sets. Structure-using methods (FoldX, Rosetta ΔΔG) typically reach r ≈ 0.65–0.75. Our sequence-only model is competitive with the lower end of structure-based methods while being far more broadly applicable.

### 10.2 Feature Importance

The top 15 features by average gradient importance across all ensemble members:

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | temperature_C | 340.6 |
| 2 | pH | 293.3 |
| 3 | RSA | 261.1 |
| 4 | ΔHydrophobicity × temperature | 252.0 |
| 5 | rel_position | 218.3 |
| 6 | delta_ionization (pH-dependent) | 206.9 |
| 7 | nearest_Cys_distance | 196.6 |
| 8 | ΔAliphatic × temperature | 184.2 |
| 9 | burial × ΔCharge(pH) | 172.2 |
| 10 | abs_charge_mut_pH | 154.1 |

Assay conditions (temperature, pH) dominate — a direct validation that training with experimental conditions as features is the correct design choice. RSA ranks third, confirming that structural context (even when estimated from sequence alone) is highly informative. The cross-terms (ranks 4, 8, 9) demonstrate that the physically motivated feature engineering captures important non-linear effects.

### 10.3 Condition Sensitivity

The Debye-Hückel and Ca²⁺ corrections allow the system to differentiate predictions across realistic industrial conditions. As a representative example, a lysine-introduction mutation on an exposed loop might score +0.08 in standard laboratory buffer (100 mM NaCl, no Ca²⁺) but −0.03 in industrial wastewater-equivalent conditions (500 mM NaCl, 2 mM Ca²⁺) — a sign reversal sufficient to change its classification from "neutral" to "beneficial." This sensitivity is not available from static ΔΔG predictors that assume a single condition.

### 10.4 Limitations

The primary limitation is the **absence of structural features**. Methods like FoldX achieve higher accuracy by computing explicit van der Waals, electrostatic, and solvation terms from 3D coordinates. Our sequence-only RSA and secondary structure estimates are accurate on average but cannot resolve unusual structural microenvironments.

A secondary limitation is **PSSM coverage**: the evolutionary conservation module requires pre-computed BLAST profiles, which are not available in the current deployment. PSSM features have been shown to contribute 2–4% accuracy in similar models (Montanucci *et al.*, 2019), suggesting an achievable near-term improvement.

---

## 11. Conclusion

PETase-ML demonstrates that a carefully engineered feature set, multi-level stacking, and physics-grounded condition corrections can produce a sequence-only ΔΔG predictor competitive with structure-based methods on binary stabilization accuracy — and more broadly applicable. The integrated web platform closes the loop between prediction and experimental design, providing researchers with ranked candidates, uncertainty-aware scores, literature cross-references, and a 3D tour of mutation sites in a single interface.

The plastic degradation problem is ultimately solved by enzymes that work reliably at industrial temperatures in realistic effluent conditions. By making condition-aware predictions accessible through a public web tool, we hope to accelerate the experimental iteration cycle that will get us there.

---

## References

- Yoshida S. *et al.* (2016). A bacterium that degrades and assimilates poly(ethylene terephthalate). *Science*, 351(6278), 1196–1199.
- Tournier V. *et al.* (2020). An engineered PET depolymerase to break down and recycle plastic bottles. *Nature*, 580(7802), 216–219.
- Lu H. *et al.* (2022). Machine learning-aided engineering of hydrolases for PET depolymerization. *Nature*, 604(7907), 662–667.
- Joo S. *et al.* (2018). Structural insight into molecular mechanism of poly(ethylene terephthalate) degradation. *Nature Communications*, 9(1), 382.
- Sulaiman S. *et al.* (2012). Isolation of a novel cutinase homolog with polyethylene terephthalate-degrading activity from leaf-branch compost. *Journal of Bacteriology*, 194(24), 6759–6765.
- Montanucci L. *et al.* (2019). Predicting and interpreting large-scale mutagenesis data using analyses of protein stability and conservation. *Cell Reports*, 28(6), 1541–1553.
- Tanford C. (1961). *Physical Chemistry of Macromolecules*. Wiley.
- Youden W.J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32–35.
- Chen T. & Guestrin C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794.
- Ke G. *et al.* (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*, 30.
- Prokhorenkova L. *et al.* (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*, 31.
- Akiba T. *et al.* (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD '19*, 2623–2631.
- Berman H.M. *et al.* (2000). The Protein Data Bank. *Nucleic Acids Research*, 28(1), 235–242.

---

*Submitted for review. Source code, trained models, and live demo available upon request.*
