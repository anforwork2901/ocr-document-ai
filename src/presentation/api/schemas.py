from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Tong tien hoa don la bao nhieu?"])

