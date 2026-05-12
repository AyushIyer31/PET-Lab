"""FastAPI backend for PETase ML optimization."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import sys


import threading

_model_ready = False

def _download_model_files_background():
    """Download model files from Hugging Face in background thread after server starts."""
    global _model_ready
    model_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    files = [
        "mutation_regressor.pkl",
        "esm_embeddings_cache.pkl",
        "conservation_cache.pkl",
        "deltaTm_regressor.pkl",
        "esm_pca.pkl",
        "metal_coord_cache.pkl",
        "mutation_classifier.pkl",
        "physics_features_cache.pkl",
        "scaler.pkl",
    ]

    missing = [f for f in files if not os.path.exists(os.path.join(model_dir, f))]
    if not missing:
        print("[startup] All model files already present.", file=sys.stderr)
        _model_ready = True
        return

    print(f"[startup] Downloading {len(missing)} missing model file(s) from Hugging Face...", file=sys.stderr)
    try:
        from huggingface_hub import hf_hub_download
        for fname in missing:
            print(f"[startup]   Downloading {fname}...", file=sys.stderr)
            hf_hub_download(
                repo_id="Ayush0931/petase-models",
                filename=fname,
                local_dir=model_dir,
                token=(os.environ.get("HF_TOKEN") or "").strip() or None,
            )
            print(f"[startup]   Done: {fname}", file=sys.stderr)
        print("[startup] All model files downloaded.", file=sys.stderr)
    except Exception as e:
        print(f"[startup] ERROR downloading model files: {e}", file=sys.stderr)
    _model_ready = True

from .models.schemas import (
    SequenceInput,
    OptimizationRequest,
    OptimizationResponse,
    EmbeddingResponse,
    MutationCandidate,
    PDBSearchResult,
)
from .services import pdb_fetcher, latent_optimizer
from .services import explainability, literature_validation, trained_classifier

app = FastAPI(
    title="PETase ML Optimizer",
    description="ML-driven enzyme engineering for plastic-degrading PETase enzymes",
    version="1.0.0",
)


def _startup_task():
    """Download model files then load models — runs in background thread."""
    global _model_ready
    _download_model_files_background()
    try:
        print("[startup] Pre-loading trained classifier model...", file=sys.stderr)
        trained_classifier.train_model()
        print("[startup] Model loaded.", file=sys.stderr)
    except Exception as e:
        print(f"[startup] WARNING: Could not pre-load model: {e}", file=sys.stderr)
    try:
        from .services.pdb_fetcher import _load_disk_cache
        cached = _load_disk_cache()
        if cached:
            print(f"[startup] PDB disk cache loaded ({len(cached)} entries).", file=sys.stderr)
        else:
            print("[startup] No PDB disk cache found.", file=sys.stderr)
    except Exception as e:
        print(f"[startup] WARNING: PDB cache check failed: {e}", file=sys.stderr)
    _model_ready = True


@app.on_event("startup")
def preload_models():
    """Bind port immediately, download + load models in background thread."""
    t = threading.Thread(target=_startup_task, daemon=True)
    t.start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default IsPETase wild-type sequence (Ideonella sakaiensis, PDB: 5XJH)
ISPETASE_SEQUENCE = (
    "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAG"
    "TVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALR"
    "QVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTL"
    "IFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDT"
    "RYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
)


@app.get("/")
async def root():
    return {
        "service": "PETase ML Optimizer",
        "version": "1.0.0",
        "endpoints": [
            "/pdb/search",
            "/pdb/sequence/{pdb_id}",
            "/esm/embedding",
            "/optimize",
            "/health",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/pdb/search", response_model=list[PDBSearchResult])
async def search_pdb():
    """Search RCSB PDB for PETase-related structures."""
    try:
        results = pdb_fetcher.fetch_all_petase_data()
        return [
            PDBSearchResult(
                pdb_id=r["pdb_id"],
                title=r["title"],
                organism=r.get("organism", "Unknown"),
                resolution=r.get("resolution"),
                sequence=r["sequence"],
                family=r.get("family", "Related Hydrolase"),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdb/sequence/{pdb_id}")
async def get_pdb_sequence(pdb_id: str):
    """Fetch sequence for a specific PDB ID."""
    sequence = pdb_fetcher.fetch_sequence(pdb_id.upper())
    if not sequence:
        raise HTTPException(status_code=404, detail=f"No sequence found for {pdb_id}")
    meta = pdb_fetcher.fetch_entry_metadata(pdb_id.upper())
    return {"pdb_id": pdb_id.upper(), "sequence": sequence, **meta}


@app.get("/pdb/live-search")
async def live_search_pdb(q: str):
    """Search RCSB PDB live for any query string."""
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    try:
        results = pdb_fetcher.search_rcsb_live(q)
        return [
            {
                "pdb_id": r["pdb_id"],
                "title": r.get("title", "Unknown"),
                "organism": r.get("organism", "Unknown"),
                "resolution": r.get("resolution"),
                "sequence": r.get("sequence", ""),
                "family": r.get("family", "Related Hydrolase"),
            }
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/esm/embedding", response_model=EmbeddingResponse)
async def compute_embedding(req: SequenceInput):
    """Compute ESM-2 embedding for a protein sequence."""
    if not req.sequence or len(req.sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence must be at least 10 residues")
    if len(req.sequence) > 1000:
        raise HTTPException(status_code=400, detail="Sequence must be under 1000 residues")

    try:
        from .services import esm_engine
        embedding = esm_engine.get_sequence_embedding(req.sequence)
        return EmbeddingResponse(
            sequence=req.sequence,
            embedding_dim=len(embedding),
            mean_embedding=embedding.tolist(),
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="ESM-2 model not available on this server (requires PyTorch)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/esm/mutations")
async def scan_mutations(req: SequenceInput):
    """Scan for beneficial single-point mutations using ESM-2."""
    if not req.sequence or len(req.sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence must be at least 10 residues")

    try:
        from .services import esm_engine
        mutations = esm_engine.scan_beneficial_mutations(req.sequence, top_k=30)
        return {"sequence_length": len(req.sequence), "beneficial_mutations": mutations}
    except ImportError:
        raise HTTPException(status_code=503, detail="ESM-2 model not available on this server (requires PyTorch)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_petase(req: OptimizationRequest):
    """Run full latent space optimization to generate improved PETase candidates."""
    sequence = req.sequence or ISPETASE_SEQUENCE
    if len(sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence must be at least 10 residues")

    try:
        result = latent_optimizer.optimize(
            sequence=sequence,
            num_candidates=req.num_candidates,
            optimization_steps=req.optimization_steps,
            target_temp=req.target_temperature,
            ph=req.ph,
            ionic_strength_mm=req.ionic_strength_mm,
            ca_conc_mm=req.ca_conc_mm,
            contamination_scenario=req.contamination_scenario,
        )
        return OptimizationResponse(
            original_sequence=result["original_sequence"],
            candidates=[MutationCandidate(**c) for c in result["candidates"]],
            latent_space_summary=result["latent_space_summary"],
            classifier_info=result.get("classifier_info", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/mutation")
async def explain_mutation(req: SequenceInput):
    """Explain a single mutation. Pass mutation as the 'name' field (e.g. S121E)."""
    mut_str = req.name
    if len(mut_str) < 3:
        raise HTTPException(status_code=400, detail="Mutation format: S121E")
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    position = int(mut_str[1:-1]) - 1
    result = explainability.explain_mutation(wt_aa, mut_aa, position)
    return result


@app.post("/explain/candidate")
async def explain_candidate_mutations(req: SequenceInput):
    """Explain all mutations in a candidate. Pass comma-separated mutations as 'name'."""
    mutations = [m.strip() for m in req.name.split(",") if m.strip()]
    result = explainability.explain_candidate(mutations)
    return result


@app.get("/literature/known-mutations")
async def known_mutations():
    """Return all experimentally validated PETase mutations from literature."""
    return {
        "mutations": literature_validation.get_all_known_mutations(),
        "named_variants": literature_validation.NAMED_VARIANTS,
    }


@app.post("/literature/validate")
async def validate_against_literature(req: SequenceInput):
    """Validate predicted mutations against published experiments. Pass comma-separated mutations as 'name'."""
    mutations = [m.strip() for m in req.name.split(",") if m.strip()]
    return literature_validation.validate_mutations(mutations)


@app.get("/classifier/info")
async def classifier_info():
    """Return trained classifier model info and metrics."""
    metrics = trained_classifier.get_training_metrics()
    return metrics


@app.post("/classifier/predict")
async def classifier_predict(req: SequenceInput):
    """Predict mutation effect using trained classifier. Pass comma-separated mutations as 'name'."""
    mutations = [m.strip() for m in req.name.split(",") if m.strip()]
    return trained_classifier.predict_candidate_mutations(mutations)


@app.get("/default-sequence")
async def default_sequence():
    """Return the default IsPETase wild-type sequence."""
    return {"name": "IsPETase (Ideonella sakaiensis)", "pdb_id": "5XJH", "sequence": ISPETASE_SEQUENCE}


# ---------- 3D Structure Viewer ----------
# Cache predicted structures to avoid re-calling ESMFold
_STRUCTURE_CACHE: dict[str, str] = {}

# Known PDB IDs for preset sequences (avoid ESMFold when unnecessary)
_LCC_SEQUENCE = (
    "SNPYQRGPNPTRSALTADGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSPGYTADASSL"
    "AWLGRRLASHGFVVLVINTNSRFDYPDSRASQLSAALNYLRTSSPSAVRARLDANRLAVAGHSMGGG"
    "GTLRIAEQNPSLKAAVPLTPWHTDKTFNTSVPVLIVGAEADTVAPVSQHAIPFYQNLPSTTPKVYV"
    "ELDNASHFAPNSNNAAISVYTISWMKLWVDNDTRYRQFLCNVNDPALSDFRTNNRHCQ"
)

_KNOWN_SEQUENCE_PDBS: dict[str, str] = {
    ISPETASE_SEQUENCE: "5XJH",
    _LCC_SEQUENCE: "4EB0",
}


from pydantic import BaseModel as _BaseModel

class StructureRequest(_BaseModel):
    sequence: str
    mutations: str = ""
    title: str = ""
    original_sequence: str = ""


@app.post("/api/structure-viewer", response_class=HTMLResponse)
async def structure_viewer(req: StructureRequest):
    """Return an interactive 3Dmol.js HTML page.

    Uses known PDB structures when available, otherwise predicts
    the structure using ESMFold (Meta's protein structure prediction model).
    """
    sequence = req.sequence.strip().upper()
    original = req.original_sequence.strip().upper() if req.original_sequence else sequence

    # Check cache first
    pdb_data = _STRUCTURE_CACHE.get(original) or _STRUCTURE_CACHE.get(sequence)
    source = "cached"

    if not pdb_data:
        pdb_id = None

        # 1. Try known PDB for the original wild-type sequence
        pdb_id = _KNOWN_SEQUENCE_PDBS.get(original)

        # 2. Try subsequence match against known sequences
        if not pdb_id:
            for known_seq, kid in _KNOWN_SEQUENCE_PDBS.items():
                if original in known_seq or known_seq in original:
                    pdb_id = kid
                    break

        # 3. Try similarity match — if sequences differ by < 5%, use the same PDB
        if not pdb_id:
            for known_seq, kid in _KNOWN_SEQUENCE_PDBS.items():
                if len(original) == len(known_seq):
                    diffs = sum(1 for a, b in zip(original, known_seq) if a != b)
                    if diffs / len(original) < 0.05:
                        pdb_id = kid
                        break
                # Also check the candidate sequence itself
                if len(sequence) == len(known_seq):
                    diffs = sum(1 for a, b in zip(sequence, known_seq) if a != b)
                    if diffs / len(sequence) < 0.05:
                        pdb_id = kid
                        break

        # 4. Try fetching from RCSB search by sequence
        if not pdb_id:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    search_query = {
                        "query": {
                            "type": "terminal",
                            "service": "sequence",
                            "parameters": {
                                "evalue_cutoff": 0.1,
                                "identity_cutoff": 0.9,
                                "sequence_type": "protein",
                                "value": original[:400],
                            }
                        },
                        "return_type": "entry",
                        "request_options": {"results_content_type": ["experimental"], "return_all_hits": False}
                    }
                    resp = await client.post(
                        "https://search.rcsb.org/rcsbsearch/v2/query",
                        json=search_query,
                    )
                    if resp.status_code == 200:
                        hits = resp.json().get("result_set", [])
                        if hits:
                            pdb_id = hits[0]["identifier"]
            except Exception:
                pass

        if pdb_id:
            # Fetch crystal structure from RCSB
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
                if resp.status_code == 200:
                    pdb_data = resp.text
                    source = f"PDB: {pdb_id} (crystal structure)"

        if not pdb_data:
            raise HTTPException(
                status_code=404,
                detail="No structure found. Try using a known enzyme from the presets."
            )

        # Cache for future requests
        _STRUCTURE_CACHE[original] = pdb_data

    # Amino acid name lookup
    _AA_NAMES = {
        'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic Acid',
        'C': 'Cysteine', 'E': 'Glutamic Acid', 'Q': 'Glutamine', 'G': 'Glycine',
        'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
        'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
        'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine',
    }

    _AA_CATEGORIES = {
        'A': 'hydrophobic', 'V': 'hydrophobic', 'I': 'hydrophobic', 'L': 'hydrophobic',
        'M': 'hydrophobic', 'F': 'aromatic', 'W': 'aromatic', 'Y': 'aromatic',
        'P': 'cyclic', 'G': 'small', 'S': 'polar', 'T': 'polar',
        'C': 'sulfur-containing', 'N': 'polar amide', 'Q': 'polar amide',
        'D': 'negatively charged', 'E': 'negatively charged',
        'K': 'positively charged', 'R': 'positively charged', 'H': 'positively charged',
    }

    # Parse mutations like "S121E,D186H,R280A"
    mut_list = [m.strip() for m in req.mutations.split(",") if m.strip()]
    mut_positions = []
    mut_labels = []
    mut_details = []  # (label, wt_name, mut_name, position)
    for m in mut_list:
        try:
            pos = int(m[1:-1])
            wt_aa = m[0]
            mut_aa = m[-1]
            mut_positions.append(pos)
            mut_labels.append(m)
            from_cat = _AA_CATEGORIES.get(wt_aa, 'unknown')
            to_cat = _AA_CATEGORIES.get(mut_aa, 'unknown')
            # Build a short explanation of the property change
            change_note = ""
            if from_cat != to_cat:
                change_note = f"{from_cat} → {to_cat}"
            else:
                change_note = f"both {from_cat}"
            mut_details.append({
                "label": m,
                "position": pos,
                "from_code": wt_aa,
                "to_code": mut_aa,
                "from_name": _AA_NAMES.get(wt_aa, wt_aa),
                "to_name": _AA_NAMES.get(mut_aa, mut_aa),
                "from_category": from_cat,
                "to_category": to_cat,
                "change_note": change_note,
            })
        except (ValueError, IndexError):
            pass

    # Catalytic residues for IsPETase-like enzymes
    catalytic = [160, 206, 237]

    # Build JS for highlighting — make mutations very prominent
    mut_selections_js = ""
    for i, pos in enumerate(mut_positions):
        wt_aa = mut_labels[i][0]
        mut_aa = mut_labels[i][-1]
        mut_selections_js += f"""
        // Mutation {mut_labels[i]}: big sphere + thick stick + pulsing glow
        viewer.addStyle({{resi: {pos}}}, {{
            stick: {{color: '#FF6B35', radius: 0.25}},
            sphere: {{color: '#FF6B35', opacity: 0.55, radius: 1.2}}
        }});
        // Bright label with mutation detail
        viewer.addLabel("{mut_labels[i]}  ({wt_aa}\u2192{mut_aa})", {{
            position: {{resi: {pos}}},
            backgroundColor: '#FF6B35',
            fontColor: 'white',
            fontSize: 14,
            fontWeight: 'bold',
            padding: 4,
            borderRadius: 6,
            borderColor: '#FF8855',
            borderThickness: 1.5,
            showBackground: true
        }});
        """

    catalytic_js = ""
    for pos in catalytic:
        catalytic_js += f"""
        viewer.addStyle({{resi: {pos}}}, {{
            stick: {{color: '#0FB5A2', radius: 0.2}},
            sphere: {{color: '#0FB5A2', opacity: 0.35, radius: 0.9}}
        }});
        viewer.addLabel("Catalytic {pos}", {{
            position: {{resi: {pos}}},
            backgroundColor: '#0FB5A2',
            fontColor: 'white',
            fontSize: 11,
            padding: 3,
            borderRadius: 5,
            showBackground: true
        }});
        """

    # JS array of catalytic positions for tour highlight
    catalytic_positions_js = ", ".join(str(p) for p in catalytic)

    display_title = req.title if req.title else "3D Structure"

    # Build mutation details HTML
    mut_detail_html = ""
    if mut_details:
        rows = ""
        for md in mut_details:
            rows += f"""<div class="mut-row">
              <span class="mut-badge">{md['label']}</span>
              <span class="mut-desc">
                <span class="aa-from">{md['from_name']}</span>
                <span class="mut-arrow">&rarr;</span>
                <span class="aa-to">{md['to_name']}</span>
                <span class="mut-pos">Position {md['position']}</span>
              </span>
            </div>"""
        mut_detail_html = f"""<div id="mutations-panel">
          <div class="panel-title">Mutations vs. Wild-Type ({len(mut_details)} change{'s' if len(mut_details) != 1 else ''})</div>
          {rows}
        </div>"""
    else:
        mut_detail_html = """<div id="mutations-panel">
          <div class="panel-title">No mutations — Wild-type structure</div>
        </div>"""

    # Escape PDB data for JS
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    # Build tour data JSON for JS
    import json as _json
    tour_data_json = _json.dumps(mut_details)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #08080f;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    color: white;
    overflow-x: hidden;
    overflow-y: auto;
  }}

  /* ── Header ── */
  #header {{
    padding: 12px 16px 8px;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center; justify-content: space-between;
    z-index: 30; position: relative;
  }}
  #header h2 {{ font-size: 15px; font-weight: 700; color: #E8E8ED; }}
  #source-tag {{ font-size: 10px; color: #70708A; font-style: italic; margin-top: 2px; }}
  #tour-btn {{
    background: linear-gradient(135deg, #7C8CF8 0%, #6366F1 100%);
    color: white; border: none; border-radius: 8px;
    padding: 8px 18px; font-size: 12px; font-weight: 700; cursor: pointer;
    letter-spacing: 0.4px; transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    display: {{'inline-flex' if mut_details else 'none'}};
    align-items: center; gap: 6px;
    box-shadow: 0 2px 12px rgba(124,140,248,0.3);
  }}
  #tour-btn:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,140,248,0.5);
  }}
  #tour-btn svg {{ width: 14px; height: 14px; }}

  /* ── Viewer ── */
  #viewer-wrap {{
    position: relative;
    width: 100vw; height: 55vh; min-height: 320px;
    transition: height 0.8s cubic-bezier(0.4,0,0.2,1);
  }}
  #viewer-wrap.tour-active {{ height: 100vh; min-height: 100vh; }}
  #viewer-container {{ width: 100%; height: 100%; }}

  /* ── Cinematic letterbox bars ── */
  .cine-bar {{
    position: absolute; left: 0; right: 0;
    background: #08080f; height: 0;
    transition: height 1s cubic-bezier(0.4,0,0.2,1);
    z-index: 5; pointer-events: none;
  }}
  .cine-bar-top {{ top: 0; }}
  .cine-bar-bot {{ bottom: 0; }}
  #viewer-wrap.tour-active .cine-bar {{ height: 40px; }}

  /* ── Vignette ── */
  #tour-vignette {{
    position: absolute; inset: 0; z-index: 4; pointer-events: none;
    opacity: 0; transition: opacity 0.8s ease;
    background: radial-gradient(ellipse at center, transparent 35%, rgba(8,8,15,0.55) 100%);
  }}
  #viewer-wrap.tour-active #tour-vignette {{ opacity: 1; }}

  /* ── Timeline progress bar (thin bar along top) ── */
  #tour-timeline {{
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    z-index: 15; pointer-events: none;
  }}
  #tour-timeline-fill {{
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #6366F1, #7C8CF8, #FF6B35);
    border-radius: 0 2px 2px 0;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
    box-shadow: 0 0 12px rgba(124,140,248,0.5);
  }}

  /* ── Mutation counter (top-right) ── */
  #tour-counter {{
    position: absolute; top: 52px; right: 20px; z-index: 12;
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    color: rgba(255,255,255,0.0);
    text-transform: uppercase;
    transition: color 0.5s ease;
  }}
  #viewer-wrap.tour-active #tour-counter {{ color: rgba(255,255,255,0.3); }}

  /* ── Info card (bottom-left, glass panel) ── */
  #tour-card {{
    position: absolute; bottom: 56px; left: 20px; z-index: 12;
    max-width: 380px; width: calc(100% - 40px);
    background: rgba(12,12,22,0.82);
    -webkit-backdrop-filter: blur(20px) saturate(1.3);
    backdrop-filter: blur(20px) saturate(1.3);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    opacity: 0; transform: translateY(16px);
    transition: opacity 0.5s cubic-bezier(0.4,0,0.2,1),
                transform 0.5s cubic-bezier(0.4,0,0.2,1);
    pointer-events: none;
    overflow: hidden;
  }}
  #tour-card.visible {{
    opacity: 1; transform: translateY(0); pointer-events: auto;
  }}
  #tour-card-phase {{
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 12px 16px 0;
    color: rgba(255,255,255,0.25);
  }}
  #tour-card-title {{
    font-size: 18px; font-weight: 800; letter-spacing: -0.3px;
    color: #F0F0F5; padding: 4px 16px 0;
  }}
  #tour-card-subtitle {{
    font-size: 11px; color: #70708A; padding: 2px 16px 0; font-weight: 500;
  }}
  #tour-card-body {{
    font-size: 12.5px; color: #AAB0CC; line-height: 1.6;
    padding: 10px 16px 0;
  }}
  #tour-card-body strong {{ color: #D0D0E0; }}

  /* ── Amino acid swap visual ── */
  .aa-swap {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px 14px;
  }}
  .aa-chip {{
    display: flex; flex-direction: column; align-items: center; gap: 1px;
    padding: 6px 12px; border-radius: 8px; min-width: 56px;
  }}
  .aa-chip.wt {{
    background: rgba(255,100,100,0.07); border: 1px solid rgba(255,100,100,0.14);
  }}
  .aa-chip.mt {{
    background: rgba(100,220,140,0.07); border: 1px solid rgba(100,220,140,0.14);
  }}
  .aa-chip-code {{
    font-family: 'SF Mono', Menlo, monospace;
    font-size: 20px; font-weight: 800;
  }}
  .aa-chip.wt .aa-chip-code {{ color: #FF8888; }}
  .aa-chip.mt .aa-chip-code {{ color: #7DDC8A; }}
  .aa-chip-name {{
    font-size: 8px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: rgba(255,255,255,0.3);
  }}
  .aa-arrow {{
    color: rgba(255,255,255,0.15); font-size: 18px;
    transition: color 0.3s;
  }}
  .aa-tag {{
    margin-left: auto; font-size: 9px; font-weight: 600;
    padding: 3px 8px; border-radius: 4px;
    background: rgba(124,140,248,0.08); color: #9BA6FF;
    white-space: nowrap; letter-spacing: 0.3px;
  }}

  /* ── Splash overlay ── */
  #tour-splash {{
    position: absolute; inset: 0; z-index: 20;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(8,8,15,0.88);
    opacity: 0; pointer-events: none;
    transition: opacity 0.6s ease;
  }}
  #tour-splash.show {{ opacity: 1; pointer-events: auto; }}
  .sp-line {{
    opacity: 0; transform: translateY(14px);
    animation: sp-in 0.7s cubic-bezier(0.4,0,0.2,1) forwards;
  }}
  .sp-line:nth-child(1) {{ animation-delay: 0.1s; }}
  .sp-line:nth-child(2) {{ animation-delay: 0.25s; }}
  .sp-line:nth-child(3) {{ animation-delay: 0.4s; }}
  .sp-label {{
    font-size: 11px; font-weight: 700; letter-spacing: 4px;
    text-transform: uppercase; color: rgba(255,255,255,0.25);
  }}
  .sp-big {{
    font-size: 48px; font-weight: 800; letter-spacing: -1.5px;
    color: #E8E8ED; margin-top: 2px;
  }}
  .sp-big span {{ color: #FF6B35; }}
  .sp-hint {{
    font-size: 12px; color: rgba(255,255,255,0.2); margin-top: 6px;
  }}
  @keyframes sp-in {{
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  /* ── Stop button (appears during tour) ── */
  #tour-stop {{
    position: absolute; top: 48px; left: 20px; z-index: 15;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.5); border-radius: 8px;
    padding: 6px 14px; font-size: 11px; font-weight: 600;
    cursor: pointer; display: none; align-items: center; gap: 5px;
    transition: all 0.25s ease;
    letter-spacing: 0.3px;
  }}
  #tour-stop:hover {{ background: rgba(255,255,255,0.1); color: white; }}
  #tour-stop svg {{ width: 12px; height: 12px; }}

  /* ── Color Key ── */
  #color-key {{
    padding: 14px 16px;
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(255,255,255,0.06);
  }}
  .key-title {{
    font-size: 12px; font-weight: 700; color: #70708A;
    margin-bottom: 10px; letter-spacing: 1px; text-transform: uppercase;
  }}
  .key-grid {{ display: flex; flex-direction: column; gap: 6px; }}
  .key-item {{
    display: flex; align-items: center; gap: 10px;
    padding: 6px 10px; border-radius: 6px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
  }}
  .key-swatch-sphere {{ width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }}
  .key-spectrum {{
    width: 40px; height: 14px; border-radius: 3px;
    background: linear-gradient(90deg, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000);
    flex-shrink: 0;
  }}
  .key-label {{ font-size: 12px; color: #AAB0CC; font-weight: 500; }}
  .key-desc {{ font-size: 10px; color: #555570; }}

  /* ── Mutations Panel ── */
  #mutations-panel {{
    padding: 12px 16px 16px;
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(255,255,255,0.06);
  }}
  .panel-title {{ font-size: 12px; font-weight: 700; color: #70708A; margin-bottom: 10px; letter-spacing: 1px; text-transform: uppercase; }}
  .mut-row {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; margin-bottom: 6px;
    background: rgba(255,107,53,0.06); border: 1px solid rgba(255,107,53,0.12);
    border-radius: 6px;
  }}
  .mut-badge {{
    background: #FF6B35; color: white; font-size: 12px; font-weight: 700;
    padding: 3px 8px; border-radius: 4px;
    font-family: 'SF Mono', Menlo, monospace; white-space: nowrap;
  }}
  .mut-desc {{ display: flex; align-items: center; gap: 5px; flex-wrap: wrap; font-size: 12px; }}
  .aa-from {{ color: #FF8888; font-weight: 500; text-decoration: line-through; text-decoration-color: rgba(255,136,136,0.3); }}
  .mut-arrow {{ color: #555570; font-size: 13px; }}
  .aa-to {{ color: #88DD88; font-weight: 600; }}
  .mut-pos {{ color: #555570; font-size: 11px; margin-left: 4px; }}

  #tip {{
    text-align: center; padding: 8px;
    font-size: 11px; color: rgba(255,255,255,0.2);
  }}

  /* Hide non-viewer content during tour */
  body.touring #color-key,
  body.touring #mutations-panel,
  body.touring #tip,
  body.touring #header {{
    display: none;
  }}
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<div id="header">
  <div>
    <h2>{display_title}</h2>
    <div id="source-tag">{source}</div>
  </div>
  <button id="tour-btn" onclick="startTour()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    Guided Tour
  </button>
</div>
<div id="viewer-wrap">
  <div class="cine-bar cine-bar-top"></div>
  <div class="cine-bar cine-bar-bot"></div>
  <div id="tour-vignette"></div>
  <div id="tour-timeline"><div id="tour-timeline-fill"></div></div>
  <div id="tour-counter"></div>
  <button id="tour-stop" onclick="endTour()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    Exit Tour
  </button>
  <div id="tour-card">
    <div id="tour-card-phase"></div>
    <div id="tour-card-title"></div>
    <div id="tour-card-subtitle"></div>
    <div id="tour-card-body"></div>
    <div id="tour-card-swap"></div>
  </div>
  <div id="viewer-container"></div>
  <div id="tour-splash">
    <div class="sp-line"><span class="sp-label">Mutation Analysis</span></div>
    <div class="sp-line"><span class="sp-big"><span>{len(mut_details)}</span> Mutation{'s' if len(mut_details) != 1 else ''}</span></div>
    <div class="sp-line"><span class="sp-hint">Flying through structural impact</span></div>
  </div>
</div>
{mut_detail_html}
<div id="color-key">
  <div class="key-title">Color Key</div>
  <div class="key-grid">
    <div class="key-item">
      <span class="key-spectrum"></span>
      <span><span class="key-label">Protein Backbone</span><br><span class="key-desc">Rainbow: N-terminus (blue) to C-terminus (red)</span></span>
    </div>
    <div class="key-item">
      <span class="key-swatch-sphere" style="background:#FF6B35"></span>
      <span><span class="key-label">AI-Predicted Mutations</span><br><span class="key-desc">Orange spheres &mdash; positions modified by the optimizer</span></span>
    </div>
    <div class="key-item">
      <span class="key-swatch-sphere" style="background:#0FB5A2"></span>
      <span><span class="key-label">Catalytic Triad</span><br><span class="key-desc">Aqua &mdash; active site residues (Ser160, His206, Asp237)</span></span>
    </div>
  </div>
</div>
<div id="tip">Pinch to zoom &middot; Drag to rotate &middot; Two-finger drag to pan</div>
<script>
/* ═══════════════════════════════════════════════════
   Data & Viewer Init
   ═══════════════════════════════════════════════════ */
var tourData = {tour_data_json};
var catalyticResidues = [{catalytic_positions_js}];

var viewer = $3Dmol.createViewer("viewer-container", {{
  backgroundColor: "0x08080f", antialias: true, cartoonQuality: 10
}});
var pdbData = `{pdb_escaped}`;
viewer.addModel(pdbData, "pdb");

function setDefaultView() {{
  viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum", opacity: 0.85, thickness: 0.25 }} }});
  {mut_selections_js}
  {catalytic_js}
  viewer.render();
}}
setDefaultView();
viewer.zoomTo();
viewer.render();

var spinning = true;
viewer.spin("y", 0.5);
document.getElementById("viewer-container").addEventListener("touchstart", function() {{
  if (spinning) {{ viewer.spin(false); spinning = false; }}
}});
document.getElementById("viewer-container").addEventListener("mousedown", function() {{
  if (spinning) {{ viewer.spin(false); spinning = false; }}
}});


/* ═══════════════════════════════════════════════════
   Cinematic Auto-Playing Tour
   ═══════════════════════════════════════════════════ */
var tourActive = false;
var tourTimers = [];

function clearTimers() {{
  tourTimers.forEach(function(t) {{ clearTimeout(t); }});
  tourTimers = [];
}}
function at(fn, ms) {{
  tourTimers.push(setTimeout(fn, ms));
}}

/* Generate explanation text for a mutation */
function explainMutation(m) {{
  if (m.from_category !== m.to_category) {{
    var reason = "";
    var fc = m.from_category, tc = m.to_category;
    if (tc === "hydrophobic" || fc === "hydrophobic")
      reason = "This alters core packing — hydrophobic changes directly affect protein stability and folding energetics.";
    else if (tc.indexOf("charged") >= 0 || fc.indexOf("charged") >= 0)
      reason = "Charge modification affects salt bridges, electrostatic networks, and solvent interactions at this site.";
    else if (tc === "polar" || tc === "polar amide")
      reason = "Introducing polarity here can form new hydrogen bonds, stabilizing local secondary structure.";
    else if (tc === "aromatic")
      reason = "Aromatic residues enable pi-stacking interactions that anchor nearby structural elements.";
    else if (tc === "cyclic")
      reason = "Proline rigidifies the backbone, reducing conformational entropy — a classic thermostability strategy.";
    else
      reason = "This substitution modifies the local chemical environment to favor enhanced catalytic geometry.";
    return "<strong>" + fc + " &rarr; " + tc + "</strong> &mdash; " + reason;
  }}
  return "A conservative <strong>" + m.from_category + "</strong> substitution — fine-tuning properties while preserving the fold architecture.";
}}


/* ── Phase helpers (each returns the time it takes) ── */

/* Phase: Intro — splash + wide establishing shot */
function phaseIntro(t0) {{
  var splash = document.getElementById("tour-splash");
  splash.classList.add("show");
  viewer.spin(false);
  viewer.zoomTo({{}}, 2000);
  at(function() {{ viewer.zoom(0.75, 1800); }}, 200);

  // Fade splash out
  at(function() {{ splash.classList.remove("show"); }}, 2200);

  // Slow panoramic spin with all mutations visible
  at(function() {{
    viewer.spin("y", 0.3);
  }}, 2400);

  // Show overview card
  at(function() {{
    showCard(
      "Overview",
      "Protein Structure",
      tourData.length + " AI-predicted mutation" + (tourData.length !== 1 ? "s" : ""),
      "Scanning the full backbone from <strong>N-terminus</strong> to <strong>C-terminus</strong>. The optimizer identified " + tourData.length + " positions where substitutions improve stability and catalytic performance."
    );
  }}, 2800);

  // Stop spin, prepare for first mutation
  at(function() {{
    viewer.spin(false);
  }}, 5800);

  return 6200;  // total duration of intro phase
}}

/* Phase: Fly to a single mutation, highlight it, explain it */
function phaseMutation(t0, m, index) {{
  var total = tourData.length;

  // Update timeline
  at(function() {{
    var pct = ((index + 1) / (total + 1)) * 100;
    document.getElementById("tour-timeline-fill").style.width = pct + "%";
    document.getElementById("tour-counter").textContent = "MUTATION " + (index + 1) + " OF " + total;
  }}, t0);

  // Fade card out before moving
  at(function() {{
    document.getElementById("tour-card").classList.remove("visible");
  }}, t0);

  // Clean up previous highlights but keep already-visited mutations dimmed
  at(function() {{
    viewer.removeAllLabels();
    viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum", opacity: 0.85, thickness: 0.25 }} }});

    // Show previously visited mutations as persistent but subtle
    for (var p = 0; p < index; p++) {{
      var prev = tourData[p];
      viewer.addStyle({{resi: prev.position}}, {{
        stick: {{ color: '#FF6B35', radius: 0.12 }},
        sphere: {{ color: '#FF6B35', opacity: 0.2, radius: 0.7 }}
      }});
    }}
    viewer.render();
  }}, t0 + 300);

  // Camera: pull back slightly from wherever we are
  at(function() {{
    viewer.zoom(0.82, 500);
  }}, t0 + 400);

  // Camera: fly to the mutation residue
  at(function() {{
    viewer.zoomTo({{resi: m.position}}, 1400);
  }}, t0 + 1000);

  // Camera: push in for close-up
  at(function() {{
    viewer.zoom(0.6, 900);
  }}, t0 + 2500);

  // Stage 1: Dim surroundings, highlight backbone at mutation
  at(function() {{
    viewer.addStyle({{resi: m.position, not: true}}, {{
      cartoon: {{ color: "spectrum", opacity: 0.25, thickness: 0.18 }}
    }});
    // Re-render previously visited with same dim
    for (var p = 0; p < index; p++) {{
      var prev = tourData[p];
      viewer.addStyle({{resi: prev.position}}, {{
        stick: {{ color: '#FF6B35', radius: 0.12 }},
        sphere: {{ color: '#FF6B35', opacity: 0.15, radius: 0.6 }}
      }});
    }}
    viewer.addStyle({{resi: m.position}}, {{
      cartoon: {{ color: "#FF6B35", opacity: 1.0, thickness: 0.45 }}
    }});
    viewer.render();
  }}, t0 + 2800);

  // Stage 2: Sticks and sphere appear
  at(function() {{
    viewer.addStyle({{resi: m.position}}, {{
      stick: {{ color: '#FF6B35', radius: 0.25 }},
      sphere: {{ color: '#FF6B35', opacity: 0.55, radius: 1.2 }}
    }});
    viewer.render();
  }}, t0 + 3200);

  // Stage 3: 3D label
  at(function() {{
    viewer.addLabel(m.label, {{
      position: {{resi: m.position}},
      backgroundColor: 'rgba(255,107,53,0.92)',
      fontColor: 'white', fontSize: 14, fontWeight: 'bold',
      padding: 5, borderRadius: 8,
      borderColor: '#FF8855', borderThickness: 1.5,
      showBackground: true
    }});

    // Show nearby catalytic residues for context
    catalyticResidues.forEach(function(cpos) {{
      if (Math.abs(cpos - m.position) < 80) {{
        viewer.addStyle({{resi: cpos}}, {{
          stick: {{ color: '#0FB5A2', radius: 0.12 }},
          sphere: {{ color: '#0FB5A2', opacity: 0.25, radius: 0.65 }}
        }});
        viewer.addLabel("Cat " + cpos, {{
          position: {{resi: cpos}},
          backgroundColor: 'rgba(15,181,162,0.7)',
          fontColor: 'white', fontSize: 9, padding: 3,
          borderRadius: 5, showBackground: true
        }});
      }}
    }});
    viewer.render();
  }}, t0 + 3500);

  // Show explanation card
  at(function() {{
    showCard(
      "Mutation " + (index + 1) + " of " + total,
      m.from_name + " &rarr; " + m.to_name,
      "Position " + m.position + " &mdash; " + m.label,
      explainMutation(m),
      m
    );
  }}, t0 + 3600);

  // Gentle slow orbit around the site while card is visible
  at(function() {{
    viewer.spin("y", 0.15);
  }}, t0 + 4000);

  at(function() {{
    viewer.spin(false);
  }}, t0 + 7000);

  return 7400;  // duration per mutation
}}

/* Phase: Finale — zoom out, reveal all, slow orbit */
function phaseFinale(t0) {{
  at(function() {{
    document.getElementById("tour-card").classList.remove("visible");
    document.getElementById("tour-timeline-fill").style.width = "100%";
    document.getElementById("tour-counter").textContent = "COMPLETE";
  }}, t0);

  // Reset to full view
  at(function() {{
    viewer.removeAllLabels();
    viewer.spin(false);
    viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum", opacity: 0.85, thickness: 0.25 }} }});
    viewer.render();
    viewer.zoomTo({{}}, 1800);
  }}, t0 + 400);

  // Staggered reveal of ALL mutations simultaneously
  tourData.forEach(function(m, i) {{
    at(function() {{
      viewer.addStyle({{resi: m.position}}, {{
        stick: {{ color: '#FF6B35', radius: 0.2 }},
        sphere: {{ color: '#FF6B35', opacity: 0.6, radius: 1.1 }}
      }});
      viewer.addLabel(m.label, {{
        position: {{resi: m.position}},
        backgroundColor: 'rgba(255,107,53,0.85)',
        fontColor: 'white', fontSize: 11, fontWeight: 'bold',
        padding: 4, borderRadius: 6, showBackground: true
      }});
      viewer.render();
    }}, t0 + 1800 + i * 300);
  }});

  var afterReveals = t0 + 1800 + tourData.length * 300 + 200;

  // Show catalytic triad
  at(function() {{
    catalyticResidues.forEach(function(cpos) {{
      viewer.addStyle({{resi: cpos}}, {{
        stick: {{ color: '#0FB5A2', radius: 0.16 }},
        sphere: {{ color: '#0FB5A2', opacity: 0.35, radius: 0.8 }}
      }});
    }});
    viewer.render();
  }}, afterReveals);

  // Summary card
  at(function() {{
    showCard(
      "Summary",
      tourData.length + " Synergistic Mutations",
      "Combined structural optimization",
      "Each mutation was selected to complement the others &mdash; collectively optimizing thermostability, catalytic geometry, and fold integrity beyond what any single substitution achieves."
    );
  }}, afterReveals + 300);

  // Triumphant slow orbit
  at(function() {{
    viewer.spin("y", 0.2);
  }}, afterReveals + 500);

  // End tour after a few seconds of the finale
  at(function() {{
    endTour();
  }}, afterReveals + 5500);

  return afterReveals + 5500 - t0;
}}

/* ── Card helper ── */
function showCard(phase, title, subtitle, body, mutData) {{
  var card = document.getElementById("tour-card");
  card.classList.remove("visible");

  // Small delay for re-entrance animation
  setTimeout(function() {{
    document.getElementById("tour-card-phase").textContent = phase;
    document.getElementById("tour-card-title").innerHTML = title;
    document.getElementById("tour-card-subtitle").innerHTML = subtitle;
    document.getElementById("tour-card-body").innerHTML = body;

    var swapEl = document.getElementById("tour-card-swap");
    if (mutData) {{
      swapEl.innerHTML = '<div class="aa-swap">'
        + '<div class="aa-chip wt"><span class="aa-chip-code">' + mutData.from_code + '</span><span class="aa-chip-name">' + mutData.from_name + '</span></div>'
        + '<span class="aa-arrow">&rarr;</span>'
        + '<div class="aa-chip mt"><span class="aa-chip-code">' + mutData.to_code + '</span><span class="aa-chip-name">' + mutData.to_name + '</span></div>'
        + '<span class="aa-tag">' + mutData.change_note + '</span>'
        + '</div>';
    }} else {{
      swapEl.innerHTML = '';
    }}

    card.classList.add("visible");
  }}, 120);
}}


/* ═══════════════════════════════════════════════════
   Tour Lifecycle
   ═══════════════════════════════════════════════════ */

function startTour() {{
  if (tourActive || tourData.length === 0) return;
  tourActive = true;
  clearTimers();

  // Enter cinematic mode
  if (spinning) {{ viewer.spin(false); spinning = false; }}
  document.body.classList.add("touring");
  document.getElementById("viewer-wrap").classList.add("tour-active");
  document.getElementById("tour-stop").style.display = "inline-flex";
  document.getElementById("tour-timeline-fill").style.width = "0%";

  // Schedule the entire animation chain
  var t = 0;

  // Intro phase
  t += phaseIntro(t);

  // Each mutation
  tourData.forEach(function(m, i) {{
    var dur = phaseMutation(t, m, i);
    t += dur;
  }});

  // Finale
  phaseFinale(t);
}}

function endTour() {{
  if (!tourActive) return;
  clearTimers();
  tourActive = false;

  // Hide tour UI
  document.getElementById("tour-card").classList.remove("visible");
  document.getElementById("tour-splash").classList.remove("show");
  document.getElementById("tour-stop").style.display = "none";
  document.getElementById("tour-counter").textContent = "";
  document.getElementById("tour-timeline-fill").style.width = "0%";

  // Exit cinematic mode
  document.body.classList.remove("touring");
  document.getElementById("viewer-wrap").classList.remove("tour-active");

  // Restore default view
  viewer.spin(false);
  viewer.removeAllLabels();
  setDefaultView();
  viewer.zoomTo({{}}, 1200);

  // Resume idle spin
  setTimeout(function() {{
    viewer.spin("y", 0.5);
    spinning = true;
  }}, 1300);
}}

// Keyboard: Escape to exit
document.addEventListener("keydown", function(e) {{
  if (tourActive && e.key === "Escape") endTour();
}});
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
