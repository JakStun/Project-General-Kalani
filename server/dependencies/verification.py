from fastapi import  HTTPException, Header
from typing import Annotated

import re

NANOID_REGEX = re.compile(r'^[A-Za-z0-9_-]{21}$')

async def verify_robot_id(x_robot_id: Annotated[str, Header()]):
    if x_robot_id is None:
        raise HTTPException(status_code=400, detail="X-Robot-ID")
    
    if not NANOID_REGEX.fullmatch(x_robot_id):
        raise HTTPException(status_code=400, detail="X-Robot-ID")