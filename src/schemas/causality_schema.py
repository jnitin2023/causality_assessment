from pydantic import BaseModel

class CausalityRequest(BaseModel):
    description: str
