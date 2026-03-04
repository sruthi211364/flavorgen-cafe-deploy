from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from flavorgen.data_loader import load_ingredients, load_drinks, compute_drink_flavor_vector
from flavorgen.semantic_model import hybrid_recommend

app = FastAPI(title="Flavorgen API")

# Load once at startup
drinks_df = load_drinks()
ingredients_df = load_ingredients()

class RecommendRequest(BaseModel):
    mode: str  # "description" or "prefs"
    query: Optional[str] = ""
    flavor_focus: List[str] = []
    sugar_pref: str = "any"
    caffeine_pref: str = "any"
    top_k: int = 5

@app.post("/recommend")
def recommend(req: RecommendRequest):
    query = req.query.strip() if req.query else ""
    if req.mode == "description" and not query:
        return {"results": []}

    results = hybrid_recommend(
        query=query if req.mode == "description" else "recommend based on preferences",
        flavor_focus=req.flavor_focus,
        sugar_pref=req.sugar_pref,
        caffeine_pref=req.caffeine_pref,
        top_k=min(req.top_k * 10, len(drinks_df)),
        alpha=0.6,
    )

    # enrich results with metadata for UI
    out = []
    for r in results[: req.top_k]:
        did = int(r["drink_id"])
        row = drinks_df[drinks_df["drink_id"] == did].iloc[0]
        out.append({
            "drink_id": did,
            "name": r["name"],
            "score": float(r["score"]),
            "type": row["type"],
            "temperature": row["temperature"],
            "sugar_level": row["sugar_level"],
            "caffeine_level": row["caffeine_level"],
            "tags": row["tags"],
        })

    return {"results": out}
