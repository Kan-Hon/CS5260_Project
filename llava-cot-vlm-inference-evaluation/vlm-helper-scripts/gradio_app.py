import torch
from PIL import Image
import re
import numpy as np
import copy
import argparse
import os
import traceback
import gradio as gr
from transformers import AutoProcessor, MllamaForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList

# --- Helper Classes (Stopping Criteria - Keep standalone or import from metrics file) ---
# Option 1: Copy them here if you want this script to be self-contained
class StopOnStrings(StoppingCriteria):
    """Stops generation when any of the specified strings are generated."""
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        ids_on_cpu = input_ids.cpu()
        generated_text = self.tokenizer.decode(ids_on_cpu[0], skip_special_tokens=True)
        for stop_string in self.stop_strings:
            if stop_string in generated_text: return True
        return False

class StopOnPeriod(StoppingCriteria):
    """Stops generation when a period is the last generated token."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        ids_on_cpu = input_ids.cpu()
        generated_text = self.tokenizer.decode(ids_on_cpu[0], skip_special_tokens=True)
        if generated_text.endswith('.'): return True
        return False

# Option 2: If metrics script is in the same directory or installable:
# from inference_metrics import StopOnStrings, StopOnPeriod

# --- LLaVA-CoT Application Class ---

class LlavaCotApp:
    def __init__(self, model_path, device="cuda", dtype_str="bfloat16", default_beam_size=2):
        """Initializes the Gradio app handler, loading the model and processor."""
        print(f"--- Initializing LlavaCotApp ---")
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.dtype = self._setup_dtype(dtype_str)
        self.default_beam_size = default_beam_size
        # Default generation kwargs for Gradio (can be overridden)
        self.generation_kwargs = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)

        print(f"Using Device: {self.device}")
        print(f"Using Torch Dtype: {self.dtype}")
        print(f"Default Beam Size: {self.default_beam_size}")

        self.model = None
        self.processor = None
        self._load_model()
        print("--- App Initialized Successfully ---")

    def _setup_device(self, device_arg):
        """Determines and validates the compute device."""
        if device_arg == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_arg == "cuda":
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        else:
            return torch.device("cpu")

    def _setup_dtype(self, dtype_str):
        """Determines the torch dtype based on input string and availability."""
        if dtype_str == "bfloat16" and self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif dtype_str == "float16" and self.device.type == 'cuda':
            return torch.float16
        else:
             # Default or if requested type not supported
             if dtype_str not in ["bfloat16", "float16"]:
                  print(f"Using default float32 dtype.")
             else:
                  print(f"WARNING: {dtype_str} requested but not supported on {self.device}. Falling back to float32.")
             return torch.float32

    def _load_model(self):
        """Loads the model and processor."""
        print(f"\n--- Loading Model and Processor from: {self.model_path} ---")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device).eval()
            print("--- Model and Processor Loaded Successfully ---")
        except Exception as e:
            print(f"FATAL ERROR: Could not load model/processor from {self.model_path}.")
            print(traceback.format_exc())
            # In Gradio context, we might want the app to show an error rather than exit
            raise RuntimeError(f"Failed to load model: {e}")


    # --- CoT Generation Logic (Adapted from metrics script) ---
    #     (Includes judge, _prepare_inputs, generate_inner_*)
    #     Make sure they use self.model, self.processor, self.device, self.generation_kwargs

    def judge(self, image, prompt, outputs, type="summary"):
        """Judges between two outputs. (Placeholder - Needs full implementation)"""
        # --- PASTE YOUR FULL JUDGE FUNCTION CODE HERE ---
        # --- Replace global references with self.* ---
        print(f"[Judge Placeholder] Judging type: {type}...")
        # Your actual implementation using self.model, self.processor, etc.
        return 0 # Placeholder: always choose first output

    def _prepare_inputs(self, prompt, image_path_or_pil):
        """Helper to load image and prepare inputs."""
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        elif isinstance(image_path_or_pil, Image.Image):
             # Gradio might pass PIL Image directly if type="pil"
             image = image_path_or_pil.convert('RGB')
        else:
            # Gradio filepath type gives string path
            raise ValueError("image_path_or_pil must be a file path (str) or a PIL Image object")

        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        return inputs, image

    # --- generate_inner_stage_beam ---
    # --- (Copy the full method from the refactored metrics script here) ---
    # --- Ensure all references use self.* ---
    def generate_inner_stage_beam(self, prompt, image_path_or_pil, beam_size=2):
        """Generates stage by stage, judging candidates after each stage."""
        inputs, image = self._prepare_inputs(prompt, image_path_or_pil)
        initial_length = len(inputs['input_ids'][0])

        stages = ['<SUMMARY>', '<CAPTION>', '<REASONING>', '<CONCLUSION>']
        end_markers = ['</SUMMARY>', '</CAPTION>', '</REASONING>', '</CONCLUSION>']
        current_beams = [copy.deepcopy(inputs['input_ids'])]

        print("--- Starting Stage Beam Generation (Gradio) ---")
        for stage_idx, (stage, end_marker) in enumerate(zip(stages, end_markers)):
            print(f"\nGenerating Stage: {stage}...")
            stage_candidates_data = []
            if not current_beams: return "Error: No beams left."

            for beam_idx, beam_input_ids in enumerate(current_beams):
                stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], self.processor.tokenizer)])
                generation_kwargs = self.generation_kwargs.copy()
                generation_kwargs.update({'stopping_criteria': stop_criteria, 'max_new_tokens': 512})
                current_inputs = {'input_ids': beam_input_ids.to(self.device), 'pixel_values': inputs['pixel_values'].to(self.device)}

                print(f"  Expanding beam {beam_idx+1}/{len(current_beams)}...")
                try:
                    temp_candidates = []
                    for cand_idx in range(beam_size):
                         if 'num_beams' in generation_kwargs: del generation_kwargs['num_beams']
                         output = self.model.generate(**current_inputs, **generation_kwargs)
                         new_generated_ids = output[0]
                         generated_text = self.processor.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
                         # print(f"    Cand {cand_idx+1} generated.") # Less verbose for Gradio
                         temp_candidates.append({'input_ids': new_generated_ids.unsqueeze(0), 'generated_text': generated_text})
                    stage_candidates_data.extend(temp_candidates)
                except Exception as e:
                     print(f"    ERROR generating candidates for stage {stage}, beam {beam_idx+1}: {e}")
                     continue

            num_candidates = len(stage_candidates_data)
            print(f"  Candidates generated for stage {stage}: {num_candidates}")
            if num_candidates == 0: return "Error: No candidates generated."

            next_beams_input_ids = []
            if num_candidates <= beam_size:
                 next_beams_input_ids = [cand['input_ids'] for cand in stage_candidates_data]
            else:
                 print(f"  Judging {num_candidates} candidates...")
                 # Simplified judging logic placeholder for Gradio (replace with actual judge call)
                 print(f"  WARNING: Using placeholder judge logic. Taking first {beam_size} candidates.")
                 next_beams_input_ids = [stage_candidates_data[i]['input_ids'] for i in range(beam_size)]
                 # --- REPLACE ABOVE WITH ACTUAL JUDGING USING self.judge ---
                 # judged_candidates = []
                 # temp_stage_candidates = stage_candidates_data.copy()
                 # while len(temp_stage_candidates) > 1 and len(judged_candidates) < beam_size:
                 #     idx1, c1 = temp_stage_candidates.pop(np.random.randint(len(temp_stage_candidates)))
                 #     idx2, c2 = temp_stage_candidates.pop(np.random.randint(len(temp_stage_candidates)))
                 #     user_prompt_match = re.search(r"USER: <image>\n(.*) ASSISTANT:", prompt, re.DOTALL)
                 #     plain_user_prompt = user_prompt_match.group(1).strip() if user_prompt_match else prompt
                 #     winner_idx = self.judge(image, plain_user_prompt, [c1['generated_text'], c2['generated_text']], type=stage[1:-1].lower())
                 #     winner = c1 if winner_idx == 0 else c2
                 #     judged_candidates.append(winner)
                 # if temp_stage_candidates and len(judged_candidates) < beam_size:
                 #     judged_candidates.extend(temp_stage_candidates)
                 # next_beams_input_ids = [cand['input_ids'] for cand in judged_candidates[:beam_size]]
                 # --- END JUDGING LOGIC EXAMPLE ---


            if not next_beams_input_ids: return "Error: No candidates survived judging."
            current_beams = next_beams_input_ids
            print(f"  Selected {len(current_beams)} beams for next stage.")

        if not current_beams: return "Error: Generation finished with no final beams."
        final_output = self.processor.decode(current_beams[0][0][initial_length:], skip_special_tokens=True)
        print(f"--- Stage Beam Generation Finished (Gradio) ---")
        return final_output


    # --- (Add generate_inner_sentence_beam and generate_inner_best_of_N methods here too if needed) ---
    # --- (Remember to copy from the refactored metrics script and adapt self.*) ---

    def process_request(self, image_input, user_prompt, generation_type="stage", beam_size_override=None):
        """
        Handles a single request from Gradio.

        Args:
            image_input: Image input from Gradio (PIL Image or filepath string).
            user_prompt (str): Text prompt from Gradio.
            generation_type (str): CoT strategy ('stage', 'sentence', 'best_of_N').
            beam_size_override (int, optional): Override default beam size.

        Returns:
            str: The generated response or an error message.
        """
        print("\n--- Processing Gradio Request ---")
        if self.model is None or self.processor is None:
             print("ERROR: Model/Processor not loaded in App.")
             return "Error: Model is not ready. Please check logs."
        if image_input is None:
            print("ERROR: No image provided.")
            return "Error: Please upload an image."
        if not user_prompt:
            print("ERROR: No prompt provided.")
            return "Error: Please enter a prompt."

        # Determine beam size
        beam_size = beam_size_override if beam_size_override is not None else self.default_beam_size

        # Prepare the LLaVA-style prompt
        llava_prompt = f"USER: <image>\n{user_prompt} ASSISTANT:"
        print(f"Type: {generation_type}, Beam Size: {beam_size}")
        print(f"Prompt: '{user_prompt[:100]}...'")

        try:
            # --- Call the appropriate internal generation method ---
            # Note: Gradio's image input type affects image_input variable type
            # If gr.Image(type="filepath"), image_input is str
            # If gr.Image(type="pil"), image_input is PIL.Image
            # The _prepare_inputs method handles both.

            start_time = time.time()
            if generation_type == "stage":
                result = self.generate_inner_stage_beam(llava_prompt, image_input, beam_size)
            # elif generation_type == "sentence":
            #     result = self.generate_inner_sentence_beam(llava_prompt, image_input, beam_size) # Add if implemented
            # elif generation_type == "best_of_N":
            #     result = self.generate_inner_best_of_N(llava_prompt, image_input, beam_size) # Add if implemented
            else:
                result = f"Error: Generation type '{generation_type}' is not implemented in this app."
                print(result)

            end_time = time.time()
            print(f"--- Gradio Request Finished (Took {end_time - start_time:.2f} seconds) ---")
            return result

        except Exception as e:
            print(f"--- ERROR during Gradio request processing ---")
            print(traceback.format_exc())
            return f"An internal error occurred during generation: {e}"


# --- Gradio Interface Setup ---

def setup_gradio_interface(app_instance):
    """Creates and returns the Gradio interface."""
    print("--- Setting up Gradio Interface ---")

    # Define the function Gradio will call. It interacts with the app_instance.
    def gradio_fn_wrapper(image_input, user_prompt, generation_type_choice):
        # Potentially get beam size from UI elements if added later
        return app_instance.process_request(image_input, user_prompt, generation_type_choice)

    # Define available CoT types for the UI
    available_cot_types = ["stage"] # Add "sentence", "best_of_N" if implemented in LlavaCotApp
    default_cot_type = "stage"

    with gr.Blocks() as demo:
        gr.Markdown("# LLaVA-CoT Inference Demo")
        gr.Markdown("Upload an image, enter a prompt, choose a strategy, and get a response.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image") # Filepath is often easier
                prompt_input = gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="Prompt")
                strategy_input = gr.Dropdown(choices=available_cot_types, value=default_cot_type, label="Generation Strategy")
                submit_button = gr.Button("Generate Response")
            with gr.Column(scale=2):
                output_textbox = gr.Textbox(lines=20, label="LLaVA-CoT Response", interactive=False)

        submit_button.click(
            fn=gradio_fn_wrapper,
            inputs=[image_input, prompt_input, strategy_input],
            outputs=output_textbox
        )

        gr.Markdown("---")
        gr.Markdown(f"Model: `{app_instance.model_path}` | Device: `{app_instance.device}` | Dtype: `{app_instance.dtype}`")

    print("--- Gradio Interface Setup Complete ---")
    return demo


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio App for LLaVA-CoT")

    # Model and App Configuration Arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained LLaVA-CoT model directory.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Data type for model ('bfloat16', 'float16', 'float32').")
    parser.add_argument("--beam_size", type=int, default=2,
                        help="Default beam size for CoT methods in the app.")

    # Gradio Launch Arguments
    parser.add_argument("--share", action="store_true",
                        help="Create a publicly shareable link for the Gradio app.")
    parser.add_argument("--server_name", type=str, default="127.0.0.1",
                        help="Interface IP address to bind to.")
    parser.add_argument("--server_port", type=int, default=7860,
                        help="Interface port to bind to.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Gradio debug mode for more detailed errors.")

    args = parser.parse_args()

    # --- Initialize the Application Handler ---
    try:
        llava_app = LlavaCotApp(
            model_path=args.model_path,
            device=args.device,
            dtype_str=args.dtype,
            default_beam_size=args.beam_size
        )
    except Exception as e:
        print(f"FATAL: Failed to initialize LlavaCotApp. Exiting. Error: {e}")
        exit(1)

    # --- Create and Launch the Gradio Interface ---
    gradio_demo = setup_gradio_interface(llava_app)

    print(f"--- Launching Gradio Demo ---")
    print(f"Access locally at: http://{args.server_name}:{args.server_port}")
    if args.share:
        print("Public share link will be generated.")

    gradio_demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug
    )