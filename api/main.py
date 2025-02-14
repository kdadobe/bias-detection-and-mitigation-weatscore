from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from bias_module.bias_module import BiasFilter  # Import bias processing logic

# Initialize FastAPI app
app = FastAPI()

# Load the bias filter model
bias_filter = BiasFilter()

# Define request model
class TextRequest(BaseModel):
    text: str

# Define response model for structured JSON output
class BiasResponse(BaseModel):
    final_statement: str
    bias_score: float  # Assuming the score is a float

@app.post("/unbias", response_model=BiasResponse)
async def process_statement(request: TextRequest = Body(...)):
    """API endpoint to detect bias in text via POST request body"""
    result, score = bias_filter.process_statement(request.text)
    return BiasResponse(final_statement=result, bias_score=score)

@app.get("/unbias", response_model=BiasResponse)
async def process_statement_get(text: str = Query(..., description="Text to be processed")):
    """API endpoint to detect bias in text via GET query parameter"""
    result, score = bias_filter.process_statement(text)
    return BiasResponse(final_statement=result, bias_score=score)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "Bias Detection API is running!"}
