from fastapi import APIRouter, UploadFile, File, Depends, Header
from typing import Annotated

from dependencies import verify_robot_id
from services import AudioService

router = APIRouter(
    prefix="/api",
    tags=["Audio"],
    dependencies=[Depends(verify_robot_id)]
)

audio_service = AudioService()

@router.post(
    "/v1/audio/post",
    summary="Robot uploads an audio file of user speaking and in return gets an audio response."
)
async def post_audio(
    x_robot_id:  Annotated[str, Header()],
    audio_file: UploadFile = File(...)
):
    return await audio_service.process_audio(x_robot_id, audio_file)