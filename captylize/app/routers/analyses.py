from fastapi import APIRouter, Depends, HTTPException
from captylize.app.dtos.analyses.request import AgeRequest, MoodRequest, NSFWRequest
from captylize.app.routers.shared import validate_image_input

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("/ages")
async def create_age_analysis(image_input: AgeRequest = Depends(validate_image_input)):
    if image_input.image_url:
        print("Analyzing from url!")
        result = {"type": "URL", "url": str(image_input.image_url)}
    elif image_input.image_file:
        print("Analyzing from file!")
        result = {"type": "FILE", "filename": image_input.image_file.filename}
    else:
        raise HTTPException(status_code=400, detail="No image provided")
    return result


@router.post("/moods")
async def create_mood_analysis(analysis: MoodRequest):
    return analysis


@router.post("/nsfw")
async def create_nsfw_analysis(analysis: NSFWRequest):
    return analysis
