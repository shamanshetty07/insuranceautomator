from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services.image_storage import ImageStorage

DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


@lru_cache
def _storage_for(root: str) -> ImageStorage:
    return ImageStorage(Path(root))


def get_storage(settings: AppSettings) -> ImageStorage:
    return _storage_for(str(settings.storage_root))


Storage = Annotated[ImageStorage, Depends(get_storage)]
