from fastapi import Depends, HTTPException, Form, File, UploadFile

from captylize.app.dtos.shared import ImageRequest, ALLOWED_IMAGE_EXTENSIONS


async def validate_image_input(
    image_url: str = Form(default=None), image_file: UploadFile = File(default=None)
) -> ImageRequest:
    if image_file and image_file.filename and image_url:
        raise HTTPException(
            status_code=400, detail="Provide either image_url or image_file, not both."
        )

    if image_file and image_file.filename:
        if not any(
            image_file.filename.lower().endswith(ext)
            for ext in ALLOWED_IMAGE_EXTENSIONS
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file type. Must be one of {ALLOWED_IMAGE_EXTENSIONS}",
            )
        return ImageRequest(image_file=image_file)
    elif image_url:
        return ImageRequest(image_url=image_url)
    else:
        raise HTTPException(
            status_code=400, detail="Either image_url or image_file must be provided."
        )