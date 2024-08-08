import logging
import mimetypes

from pydantic import BaseModel, ValidationError
from fastapi import UploadFile, HTTPException
from typing import Type

from .config import audio_types, model_settings, InferenceConfig

logger = logging.getLogger(__name__)


def process_params(parameters: dict[str, any]) -> Type[BaseModel]:
    default_fields = InferenceConfig.model_fields
    unsupported = [k for k in parameters if k not in default_fields]

    try:
        parameters = InferenceConfig(**parameters)
    except ValidationError as e:
        logger.error(f"Error validating parameters: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Error validating parameters: {e}"
        )

    if parameters.assisted:
        if not model_settings.assistant_model:
            logger.error("Assisted generation is on but to assistant model was provided")
            raise HTTPException(
                status_code=400,
                detail="Assisted generation is on but to assistant model was provided"
            )
        if parameters.batch_size > 1:
            logger.error("Batch size must be 1 when assisted generation is on")
            raise HTTPException(
                status_code=400,
                detail="Batch size must be 1 when assisted generation is on"
            )

    if unsupported:
        logger.warning(f"parameters are not supported and will be ignored: {unsupported}")

    return parameters


async def validate_file(file: UploadFile) -> bytes:
    content_type = file.content_type
    if not content_type:
        content_type = mimetypes.guess_type(file.filename)[0]
        logger.warning(f"content type was not provided, guessed as {content_type}")

    if content_type not in audio_types:
        logger.error(f"File type {file.content_type} not supported")
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not supported"
        )

    return await file.read()