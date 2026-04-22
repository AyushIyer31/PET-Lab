from pydantic import BaseModel
from typing import Optional


class SequenceInput(BaseModel):
    sequence: str
    name: str = "query_sequence"


class OptimizationRequest(BaseModel):
    sequence: str
    num_candidates: int = 10
    optimization_steps: int = 50
    target_temperature: float = 60.0
    ph: float = 8.0
    # Ionic strength (mM NaCl) — Debye-Hückel electrostatic correction
    # Typical range: 50 mM (low salt, lab) → 500 mM (industrial process water)
    ionic_strength_mm: float = 100.0
    # Ca²⁺ concentration (mM CaCl₂) — binding thermodynamics correction
    # Relevant for LCC, TfCut2, Cut190 which have Ca²⁺ binding sites
    ca_conc_mm: float = 0.0
    contamination_scenario: str = "lab"


class MutationCandidate(BaseModel):
    rank: int
    sequence: str
    mutations: list[str]
    predicted_stability_score: float
    predicted_activity_score: float
    combined_score: float
    # Explainability
    explanations: list[dict] = []
    overall_strategy: str = "balanced"
    # Literature validation
    literature_validation: dict = {}
    # Classifier prediction
    classifier_prediction: dict = {}
    # Chemical robustness (ESM-2 evolutionary fitness proxy)
    esm_robustness: float = 0.5
    esm_robustness_source: str = "classifier confidence (ESM-2 unavailable)"
    # Condition metadata
    ph_used: float = 8.0
    ph_adjustment_factor: float = 1.0
    # ΔTm: predicted change in melting temperature (°C) from ThermoMutDB-trained regressor
    # Positive values indicate the mutation increases thermal stability
    predicted_dtm: Optional[float] = None
    # Condition metadata — echoed back for client display
    ionic_strength_mm: float = 100.0
    ca_conc_mm: float = 0.0
    ionic_strength_correction: float = 0.0   # net Debye-Hückel correction (score units)
    ca_correction: float = 0.0               # net Ca²⁺ binding correction (score units)


class OptimizationResponse(BaseModel):
    original_sequence: str
    candidates: list[MutationCandidate]
    latent_space_summary: dict
    classifier_info: dict = {}


class EmbeddingResponse(BaseModel):
    sequence: str
    embedding_dim: int
    mean_embedding: list[float]


class PDBSearchResult(BaseModel):
    pdb_id: str
    title: str
    organism: str
    resolution: float | None
    sequence: str
    family: str = "Related Hydrolase"
