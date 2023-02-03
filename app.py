from setfit import SetFitModel

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = SetFitModel.from_pretrained("ilhkn/mount2_model")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    # prompt is a list 
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    preds= model(prompt)
    result = dict(zip(prompt, preds))
    # Return the results as a dictionary
    return result
