"""Model configuration endpoint (contract v1).

The frontend never assumes a provider — it reads this endpoint to learn
the active provider, the model names, and whether a fallback is in effect.
"""

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..schemas import ModelConfigOut
from ..services.llm import provider_info

router = APIRouter(tags=["model"])


@router.get("/config/model", response_model=ModelConfigOut, summary="Active model configuration")
def get_model_config(settings: Settings = Depends(get_settings)) -> ModelConfigOut:
    return ModelConfigOut(**provider_info(settings))
