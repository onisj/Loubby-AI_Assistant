from pydantic import BaseModel

class Query(BaseModel):
    text: str

class Feedback(BaseModel):
    comment: str
