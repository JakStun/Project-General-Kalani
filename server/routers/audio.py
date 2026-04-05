from fastapi import APIRouter, UploadFile, File, Depends
from typing import List, Optional, Annotated

from dependencies import verify_robot_id
from services import AudioService

router = APIRouter(
    prefix="/api",
    tags=["Audio"],
    dependencies=[Depends(verify_robot_id)]
)

@router.post(
    "/v1/audio/post",
    summary="Robot uploads an audio file of user speaking and in return gets an audio response."
)
async def post_audio(
    service: Annotated[AudioService, Depends(AudioService)],
    audio_file: UploadFile = File(...)
):
    return await service.process_audio(audio_file)