from fastapi import APIRouter

router = APIRouter(
    tags=["items"], responses={404: {"description": "Not found"}}
)


@router.get("/items")
async def read_items():
    return {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}
