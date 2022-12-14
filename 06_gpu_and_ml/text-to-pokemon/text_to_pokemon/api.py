from fastapi import FastAPI

from .main import create_pokemon_cards

web_app = FastAPI()


@web_app.get("/api/status/{call_id}")
async def poll_status(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0.1)
        return dict(
            finished=True,
            cards=result,
        )
    except TimeoutError:
        return dict(finished=False)
    except Exception:
        return dict(error="unknown job processing error")


@web_app.get("/api/create")
async def create_pokemon_job(prompt: str):
    call = create_pokemon_cards.spawn(prompt)
    return {"call_id": call.object_id}
