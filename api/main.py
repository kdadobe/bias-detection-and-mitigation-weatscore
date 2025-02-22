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
    complete_statement: str
    initial_bias_score: float
    post_processed_statement: str
    final_statement: str
    final_score: float

@app.post("/unbias", response_model=BiasResponse)
async def process_statement(request: TextRequest = Body(...)):
    """API endpoint to detect bias in text via POST request body"""
    r1,r2,r3,r4,r5 = bias_filter.process_statement(request.text)
    return BiasResponse(complete_statement=r1, initial_bias_score=r2, post_processed_statement=r3, final_statement=r4, final_score=r5)

@app.get("/unbias", response_model=BiasResponse)
async def process_statement_get(text: str = Query(..., description="Text to be processed")):
    """API endpoint to detect bias in text via GET query parameter"""
    r1,r2,r3,r4,r5 = bias_filter.process_statement(text)
    return BiasResponse(complete_statement=r1, initial_bias_score=r2, post_processed_statement=r3, final_statement=r4, final_score=r5)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "Bias Detection API is running!"}

# --------------- Gradio Integration ----------------

# Define a function for Gradio UI
def gradio_unbias(text: str):
    r1,r2,r3,r4,r5 = bias_filter.process_statement(text)
    return r1,r2,r3,r4,r5

# Create Gradio interface
gradio_app = gr.Interface(
    fn=gradio_unbias, 
    inputs=gr.Textbox(label="Enter a masked statement with [MASK] to predict the word by BERT model"),  # Input with a label
    outputs=[
        gr.Textbox(label="Sentence after BERT predicted the word for [MASK]"), 
        gr.Number(label="Initial Bias Score"), 
        gr.Textbox(label="Sentence after post processing filters are applied"), 
        gr.Textbox(label="Final Sentence after Google Flan-t5 processing"), 
        gr.Number(label="Final Bias Score")
    ], 
    title="Bias Detection",
    description="Enter a statement to analyze its bias and get a gender-neutral version."
)


# Mount Gradio inside FastAPI at `/gradio`
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
