from fastapi import APIRouter
from captylize.app.dtos.analyses.request import AgeRequest, MoodRequest, NSFWRequest

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("/ages")
async def create_age_analysis(analysis: AgeRequest):
    return analysis


@router.post("/moods")
async def create_mood_analysis(analysis: MoodRequest):
    return analysis


@router.post("/nsfw")
async def create_nsfw_analysis(analysis: NSFWRequest):
    return analysis
