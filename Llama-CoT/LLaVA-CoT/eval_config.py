from vlmeval.config import BaseConfig

class EvalConfig(BaseConfig):
    # Specify the model to use
    model = 'llama_vision'
    model_path = 'Xkev/Llama-3.2V-11B-cot'
    
    # Specify the datasets to evaluate on
    datasets = ['MMStar']  # Choose datasets as needed
    
    # Specify the output directory
    output_dir = './results'
    
    # Additional parameters
    temperature = 0.6
    top_p = 0.9
    max_new_tokens = 2048