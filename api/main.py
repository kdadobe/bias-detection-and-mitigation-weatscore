from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
import gradio as gr
from bias_module.bias_module import BiasFilter  # Import bias processing logic
import uvicorn

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

# --------------- Gradio Integration ----------------

# Define a function for Gradio UI
def gradio_unbias(text: str):
    result, score = bias_filter.process_statement(text)
    return result, score

# Create Gradio interface
gradio_app = gr.Interface(
    fn=gradio_unbias, 
    inputs=gr.Textbox(label="Enter a masked statement with [MASK] for BERT to predict the word"),
    outputs=[gr.Textbox(label="Final Statement"), gr.Number(label="Bias Score")], 
    title="Bias Detection",
    description="Enter a statement to analyze its bias score."
)

# Mount Gradio inside FastAPI at `/gradio`
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
