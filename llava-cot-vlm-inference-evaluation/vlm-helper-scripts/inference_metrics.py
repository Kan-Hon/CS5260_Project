import torch
from PIL import Image
import requests
from io import BytesIO
import os
import time
import argparse
import traceback
import re
import numpy as np
import copy
from transformers import AutoProcessor, AutoModelForCausalLM, MllamaForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList
from fvcore.nn import FlopCountAnalysis # Keep FlopCountAnalysis for now

# --- Helper Classes (Stopping Criteria - Can remain standalone) ---

class StopOnStrings(StoppingCriteria):
    """Stops generation when any of the specified strings are generated."""
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        # Pre-tokenize stop strings for potentially faster checks (optional)
        # self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

    def __call__(self, input_ids, scores, **kwargs):
        ids_on_cpu = input_ids.cpu()
        generated_text = self.tokenizer.decode(ids_on_cpu[0], skip_special_tokens=True)
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                # print(f"DEBUG: Stopping on string: {stop_string}") # Optional debug
                return True
        return False

class StopOnPeriod(StoppingCriteria):
    """Stops generation when a period is the last generated token."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        ids_on_cpu = input_ids.cpu()
        # Decode only the last few tokens for efficiency if needed, but full decode is safer
        generated_text = self.tokenizer.decode(ids_on_cpu[0], skip_special_tokens=True)
        if generated_text.endswith('.'):
            # print("DEBUG: Stopping on period.") # Optional debug
            return True
        return False

# --- Main Inference Runner Class ---

class LlavaCotRunner:
    def __init__(self, model_path, device="cuda", dtype_str="bfloat16", generation_kwargs=None):
        """
        Initializes the runner, loads the model and processor.

        Args:
            model_path (str): Path to the LLaVA-CoT model directory or file.
            device (str): Device to run on ('cuda' or 'cpu').
            dtype_str (str): Desired data type ('bfloat16', 'float16', 'float32').
            generation_kwargs (dict, optional): Default kwargs for model.generate().
        """
        print(f"--- Initializing LlavaCotRunner ---")
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.dtype = self._setup_dtype(dtype_str)
        self.generation_kwargs = generation_kwargs if generation_kwargs else \
                                  dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)

        print(f"Using Device: {self.device}")
        print(f"Using Torch Dtype: {self.dtype}")
        print(f"Default Generation Kwargs: {self.generation_kwargs}")

        self.model = None
        self.processor = None
        self._load_model()
        print("--- Runner Initialized Successfully ---")

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
        elif dtype_str not in ["bfloat16", "float16"]:
             print(f"Using default float32 dtype.")
             return torch.float32
        else:
             print(f"WARNING: {dtype_str} requested but not supported on {self.device}. Falling back to float32.")
             return torch.float32


    def _load_model(self):
        """Loads the model and processor."""
        print(f"\n--- Loading Model and Processor from: {self.model_path} ---")
        try:
            # Use AutoProcessor for flexibility
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            # Use specific class if known, otherwise AutoModelForCausalLM might work for some LLaVA variants
            # MllamaForConditionalGeneration seems specific to your setup
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True, # Good practice for large models
                trust_remote_code=True # Often needed for custom models
            ).to(self.device).eval() # Set to evaluation mode
            print("--- Model and Processor Loaded Successfully ---")

        except Exception as e:
            print(f"FATAL ERROR: Could not load model/processor from {self.model_path}.")
            print(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {e}")


    def get_model_size(self):
        """Calculates parameter count and disk size."""
        if not self.model:
            print("ERROR: Model not loaded. Cannot get size.")
            return None, None

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        total_size_bytes = 0
        path_to_check = self.model_path # Use the path stored during init
        if os.path.isdir(path_to_check):
            for dirpath, _, filenames in os.walk(path_to_check):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    # Avoid double counting / symlinks and hidden files
                    if os.path.isfile(fp) and not os.path.islink(fp) and not f.startswith('.'):
                        total_size_bytes += os.path.getsize(fp)
        elif os.path.isfile(path_to_check):
             total_size_bytes += os.path.getsize(path_to_check)
        else:
             print(f"WARNING: Could not determine disk size. Path not found or invalid: {path_to_check}")

        total_size_gb = total_size_bytes / (1024 * 1024 * 1024) if total_size_bytes > 0 else 0

        print("--- Model Size ---")
        print(f"  Model Path: {self.model_path}")
        print(f"  Total Parameters: {total_params / 1e9:.2f} B")
        print(f"  Trainable Parameters: {trainable_params / 1e6:.2f} M") # Usually 0 for inference models
        print(f"  Approx. Disk Size: {total_size_gb:.2f} GB")
        print("------------------")
        return total_params, total_size_gb

    def estimate_flops(self, sample_image_url=None, sample_prompt=None):
        """Estimates FLOPs for a single forward pass using a sample image."""
        if not self.model or not self.processor:
            print("ERROR: Model or processor not loaded. Cannot estimate FLOPs.")
            return None

        print("--- Estimating FLOPs (Forward Pass Estimate) ---")
        if sample_image_url is None:
             sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg"
        if sample_prompt is None:
            sample_prompt = "Describe the image"

        formatted_prompt = f"USER: <image>\n{sample_prompt} ASSISTANT:" # Adapt if your template differs
        total_flops_g = None

        try:
            print(f"  Loading sample image from: {sample_image_url}")
            response = requests.get(sample_image_url, timeout=10)
            response.raise_for_status()
            sample_img = Image.open(BytesIO(response.content)).convert("RGB")

            # Prepare inputs using the class's processor, device, and dtype
            inputs = self.processor(text=formatted_prompt, images=sample_img, return_tensors="pt").to(self.device, self.dtype)

            # Prepare arguments for FlopCountAnalysis
            # Adjust keys based on your specific model's forward signature if needed
            forward_args_dict = {
                "input_ids": inputs.get("input_ids"),
                "attention_mask": inputs.get("attention_mask"),
                "pixel_values": inputs.get("pixel_values")
            }
            # Filter out None values and ensure required inputs are present
            forward_args_dict = {k: v for k, v in forward_args_dict.items() if v is not None}

            if "input_ids" not in forward_args_dict or "pixel_values" not in forward_args_dict:
                 print("  Skipping FLOPs estimation: Could not prepare valid sample inputs (missing input_ids or pixel_values).")
                 return None

            print(f"  Analyzing forward pass with input shapes:")
            for k, v in forward_args_dict.items(): print(f"    {k}: {v.shape}")

            # Use the tuple of values for FlopCountAnalysis
            flops = FlopCountAnalysis(self.model, tuple(forward_args_dict.values()))
            total_flops = flops.total()
            total_flops_g = total_flops / 1e9
            print(f"  Estimated GFLOPs (forward pass): {total_flops_g:.2f} GFLOPs")
            # print(flop_count_table(flops)) # Optional: Print detailed table
            print("  NOTE: This is an estimate for one forward pass only (encoder + decoder step), not model.generate().")

        except NotImplementedError as e:
             print(f"  FLOPs estimation failed: fvcore does not support an operation in this model. Error: {e}")
        except Exception as e:
            print(f"  Error during FLOPs estimation: {e}")
            print(traceback.format_exc()) # Print stack trace for debugging
        finally:
             print("-----------------------------------------------")
             return total_flops_g

    # --- LLaVA-CoT Generation Methods ---

    def judge(self, image, prompt, outputs, type="summary"):
        """
        Judges between two outputs using the loaded model. (Placeholder - Needs full implementation)

        Args:
            image (PIL.Image): The input image.
            prompt (str): The original prompt.
            outputs (list[str]): A list containing two generated text outputs to compare.
            type (str): The type of judgment (e.g., "summary", "caption", "reasoning", "all", "sentence").

        Returns:
            int: 0 if the first output is better, 1 if the second output is better.
        """
        # --- PASTE YOUR FULL JUDGE FUNCTION CODE HERE ---
        # --- Replace global references with self.model, self.processor, self.device, self.generation_kwargs ---
        # --- Ensure image is correctly passed or processed if needed ---

        print(f"[Judge Placeholder] Judging type: {type} for prompt: '{prompt[:50]}...'")
        print(f"  Output 1: '{outputs[0][:50]}...'")
        print(f"  Output 2: '{outputs[1][:50]}...'")
        # This placeholder always chooses the first candidate.
        # Your actual implementation will involve formatting a prompt for the model
        # to compare outputs and running self.model.generate().
        # Example structure (adapt heavily based on your actual judge logic):
        # judge_prompt = f"Compare these two responses:\nResponse 1: {outputs[0]}\nResponse 2: {outputs[1]}\nWhich is better?"
        # judge_inputs = self.processor(...) # Prepare inputs with image + judge_prompt
        # judge_output_ids = self.model.generate(**judge_inputs, **self.generation_kwargs)
        # judge_response = self.processor.decode(...)
        # best_index = 0 if "1" in judge_response else 1 # Simplified logic
        best_index = 0 # Placeholder
        print(f"[Judge Placeholder] Choosing output {best_index + 1} by default.")
        return best_index


    def _prepare_inputs(self, prompt, image_path_or_pil):
        """Helper to load image and prepare inputs."""
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        elif isinstance(image_path_or_pil, Image.Image):
            image = image_path_or_pil.convert('RGB')
        else:
            raise ValueError("image_path_or_pil must be a file path (str) or a PIL Image object")

        # Assumes LLaVA chat template structure. Adapt if needed.
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        return inputs, image # Return image for judge function if needed


    def generate_inner_best_of_N(self, prompt, image_path_or_pil, beam_size=2):
        """Generates N candidates and judges to find the best."""
        inputs, image = self._prepare_inputs(prompt, image_path_or_pil)
        initial_length = len(inputs['input_ids'][0])
        input_ids_base = copy.deepcopy(inputs['input_ids']) # Keep original prompt IDs

        stop_criteria = StoppingCriteriaList([StopOnStrings(['</CONCLUSION>'], self.processor.tokenizer)])
        candidates = []

        print(f"Generating {beam_size} candidates for Best-of-N...")
        for i in range(beam_size):
            generation_kwargs = self.generation_kwargs.copy()
            generation_kwargs.update({'stopping_criteria': stop_criteria})
            # Note: 'inputs' here should only contain the prompt for each generation run
            # If model uses past_key_values, this is efficient. If not, it re-processes.
            # For simplicity, we pass the full input each time.
            current_inputs = {'input_ids': input_ids_base.to(self.device),
                              'pixel_values': inputs['pixel_values'].to(self.device)} # Pass pixels explicitly

            output = self.model.generate(**current_inputs, **generation_kwargs)
            new_generated_ids = output[0] # Full sequence
            generated_text = self.processor.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
            print(f"  Candidate {i+1}: {generated_text[:100]}...") # Debug print
            candidates.append({
                'input_ids': new_generated_ids.unsqueeze(0), # Keep full sequence with batch dim
                'generated_text': generated_text,
            })

        print(f"Judging {len(candidates)} candidates...")
        while len(candidates) > 1:
            # Simple pairwise tournament for judging
            idx1 = np.random.randint(len(candidates))
            candidate1 = candidates.pop(idx1)
            idx2 = np.random.randint(len(candidates))
            candidate2 = candidates.pop(idx2)

            outputs_to_judge = [candidate1['generated_text'], candidate2['generated_text']]
            # Use the original prompt (not the LLaVA formatted one) for the judge if needed
            # Extract original user prompt if judge function requires it
            user_prompt_match = re.search(r"USER: <image>\n(.*) ASSISTANT:", prompt, re.DOTALL)
            plain_user_prompt = user_prompt_match.group(1).strip() if user_prompt_match else prompt

            best_index = self.judge(image, plain_user_prompt, outputs_to_judge, type="all")

            winner = candidate1 if best_index == 0 else candidate2
            candidates.append(winner) # Add winner back

        final_output = candidates[0]['generated_text']
        print("--- Best-of-N Generation Finished ---")
        return final_output


    def generate_inner_sentence_beam(self, prompt, image_path_or_pil, beam_size=2):
        """Generates sentence by sentence, judging at each step."""
        inputs, image = self._prepare_inputs(prompt, image_path_or_pil)
        initial_length = len(inputs['input_ids'][0])
        current_beams = [copy.deepcopy(inputs['input_ids'])] # List of active beams (input_ids tensors)

        print("--- Starting Sentence Beam Generation ---")
        while True:
            all_candidates_data = [] # Candidates from all current beams for this step

            if not current_beams:
                 print("ERROR: No beams left in sentence beam search.")
                 return "Error: Generation failed, no beams."

            # Check if any beam has already finished
            beam_finished = False
            decoded_beam_texts = [self.processor.decode(beam[0][initial_length:], skip_special_tokens=True) for beam in current_beams]
            if any("</CONCLUSION>" in text for text in decoded_beam_texts):
                 # Find the first finished beam (simplistic, could choose best)
                 for i, text in enumerate(decoded_beam_texts):
                       if "</CONCLUSION>" in text:
                           print("--- Sentence Beam Generation Finished (Found </CONCLUSION>) ---")
                           return text
                 # Should not happen if check passes, but as fallback:
                 print("WARNING: Inconsistent state in sentence beam finish check.")
                 return decoded_beam_texts[0] # Return first beam's text

            print(f"\nGenerating next sentence for {len(current_beams)} beams...")
            for beam_idx, beam_input_ids in enumerate(current_beams):
                # Stop at period or conclusion marker
                stop_criteria = StoppingCriteriaList([
                    StopOnPeriod(self.processor.tokenizer),
                    StopOnStrings(["</CONCLUSION>"], self.processor.tokenizer)
                ])
                generation_kwargs = self.generation_kwargs.copy()
                generation_kwargs.update({
                    'stopping_criteria': stop_criteria,
                    'max_new_tokens': 128 # Limit sentence length
                })

                # Prepare inputs for this beam
                current_inputs = {'input_ids': beam_input_ids.to(self.device)}
                # Crucially, pass pixel_values if the model doesn't cache them across generate calls implicitly
                # Assuming MllamaForConditionalGeneration might need them explicitly if not using past_key_values
                current_inputs['pixel_values'] = inputs['pixel_values'].to(self.device)

                beam_candidates = []
                print(f"  Expanding beam {beam_idx+1}...")
                for cand_idx in range(beam_size): # Generate 'beam_size' continuations for this beam
                    # Avoid nested beam search if generate uses it internally
                    if 'num_beams' in generation_kwargs: del generation_kwargs['num_beams']

                    try:
                        output = self.model.generate(**current_inputs, **generation_kwargs)
                        new_generated_ids = output[0] # Full sequence
                        generated_text = self.processor.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
                        print(f"    Candidate {cand_idx+1}: ...{generated_text[-50:]}") # Show end of text
                        beam_candidates.append({
                            'input_ids': new_generated_ids.unsqueeze(0), # Add batch dim
                            'generated_text': generated_text,
                        })
                    except Exception as e:
                        print(f"    ERROR generating candidate for beam {beam_idx+1}: {e}")
                        continue # Skip this candidate generation

                all_candidates_data.extend(beam_candidates)

            if not all_candidates_data:
                print("ERROR: No candidates generated in sentence beam step.")
                # Return the best from the previous step if possible
                if current_beams:
                    return self.processor.decode(current_beams[0][0][initial_length:], skip_special_tokens=True)
                else:
                    return "Error: Generation failed, no candidates."

            print(f"Judging {len(all_candidates_data)} candidates for next sentence...")
            # Prune all candidates down to the top 'beam_size' beams globally
            next_beams_input_ids = []
            while len(all_candidates_data) > 1 and len(next_beams_input_ids) < beam_size:
                 # Simple pairwise tournament (can be improved)
                 idx1 = np.random.randint(len(all_candidates_data))
                 c1 = all_candidates_data.pop(idx1)
                 idx2 = np.random.randint(len(all_candidates_data))
                 c2 = all_candidates_data.pop(idx2)

                 outputs_to_judge = [c1['generated_text'], c2['generated_text']]
                 # Extract original user prompt if judge function requires it
                 user_prompt_match = re.search(r"USER: <image>\n(.*) ASSISTANT:", prompt, re.DOTALL)
                 plain_user_prompt = user_prompt_match.group(1).strip() if user_prompt_match else prompt

                 best_index = self.judge(image, plain_user_prompt, outputs_to_judge, type="sentence")
                 winner = c1 if best_index == 0 else c2
                 next_beams_input_ids.append(winner['input_ids']) # Keep the input_ids tensor of the winner

            # Add any remaining candidates if the loop finished early
            if len(all_candidates_data) > 0 and len(next_beams_input_ids) < beam_size:
                next_beams_input_ids.extend([cand['input_ids'] for cand in all_candidates_data])

            current_beams = next_beams_input_ids[:beam_size] # Update beams for the next iteration

            if not current_beams:
                 print("ERROR: No beams survived pruning in sentence beam.")
                 return "Error: Generation failed during pruning."

            print(f"Selected {len(current_beams)} beams for the next sentence step.")


    def generate_inner_stage_beam(self, prompt, image_path_or_pil, beam_size=2):
        """Generates stage by stage, judging candidates after each stage."""
        inputs, image = self._prepare_inputs(prompt, image_path_or_pil)
        initial_length = len(inputs['input_ids'][0])

        stages = ['<SUMMARY>', '<CAPTION>', '<REASONING>', '<CONCLUSION>']
        end_markers = ['</SUMMARY>', '</CAPTION>', '</REASONING>', '</CONCLUSION>']

        current_beams = [copy.deepcopy(inputs['input_ids'])] # List of active beam tensors

        print("--- Starting Stage Beam Generation ---")
        for stage_idx, (stage, end_marker) in enumerate(zip(stages, end_markers)):
            print(f"\nGenerating Stage: {stage} (End Marker: {end_marker})")
            stage_candidates_data = [] # Store {'input_ids': tensor, 'generated_text': str} for this stage

            if not current_beams:
                print(f"ERROR: No beams left to expand for stage {stage}. Stopping.")
                # Attempt to return the best result from the previous stage if possible
                # This part is tricky - need to track the best beam *before* it became empty
                return "Error: Generation failed, no beams left for current stage."


            # Expand each current beam
            print(f"Expanding {len(current_beams)} beams for stage {stage}...")
            for beam_idx, beam_input_ids in enumerate(current_beams):
                stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], self.processor.tokenizer)])
                generation_kwargs = self.generation_kwargs.copy()
                generation_kwargs.update({
                    'stopping_criteria': stop_criteria,
                    'max_new_tokens': 512 # Limit tokens per stage
                })

                # Prepare input for this specific beam
                current_inputs = {'input_ids': beam_input_ids.to(self.device)}
                # Pass pixel_values explicitly if model requires them each time
                current_inputs['pixel_values'] = inputs['pixel_values'].to(self.device)


                print(f"  Expanding beam {beam_idx+1}/{len(current_beams)}...")
                try:
                    # Generate multiple candidates from this beam
                    # Simpler loop approach instead of relying on num_return_sequences which might interact poorly
                    temp_candidates = []
                    for cand_idx in range(beam_size): # Generate 'beam_size' continuations
                         if 'num_beams' in generation_kwargs: del generation_kwargs['num_beams']

                         output = self.model.generate(**current_inputs, **generation_kwargs)
                         new_generated_ids = output[0] # Full sequence
                         generated_text = self.processor.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
                         print(f"    Candidate {cand_idx+1} generated.") # Minimal print
                         temp_candidates.append({
                             'input_ids': new_generated_ids.unsqueeze(0), # Add batch dim
                             'generated_text': generated_text,
                         })
                    stage_candidates_data.extend(temp_candidates)
                    print(f"    Generated {len(temp_candidates)} candidates from this beam.")

                except Exception as e:
                     print(f"    ERROR generating candidates for stage {stage}, beam {beam_idx+1}. Error: {e}")
                     print(f"    Input IDs shape: {beam_input_ids.shape}")
                     continue # Skip this beam if generation fails


            # Prune candidates from all expanded beams using the judge function
            num_candidates = len(stage_candidates_data)
            print(f"  Total candidates generated for stage {stage}: {num_candidates}")
            if num_candidates == 0:
                 print(f"ERROR: No candidates generated for stage {stage}. Stopping.")
                 # Need a strategy to return best from previous stage or indicate error
                 return "Error: Generation failed, no candidates generated for stage."

            next_beams_input_ids = []
            if num_candidates <= beam_size:
                 print(f"  Fewer candidates ({num_candidates}) than beam size ({beam_size}), keeping all.")
                 next_beams_input_ids = [cand['input_ids'] for cand in stage_candidates_data]
            else:
                 print(f"  Judging {num_candidates} candidates down to {beam_size} for stage {stage}...")
                 # Use judge function (replace placeholder logic)
                 judged_candidates = [] # Store winners

                 # Pairwise tournament judging
                 temp_stage_candidates = stage_candidates_data.copy() # Work on a copy
                 while len(temp_stage_candidates) > 1 and len(judged_candidates) < beam_size:
                    idx1 = np.random.randint(len(temp_stage_candidates))
                    c1 = temp_stage_candidates.pop(idx1)
                    idx2 = np.random.randint(len(temp_stage_candidates))
                    c2 = temp_stage_candidates.pop(idx2)

                    outputs_to_judge = [c1['generated_text'], c2['generated_text']]
                    stage_type = stage[1:-1].lower() # Get "summary", "caption", etc.
                    # Extract original user prompt if judge function requires it
                    user_prompt_match = re.search(r"USER: <image>\n(.*) ASSISTANT:", prompt, re.DOTALL)
                    plain_user_prompt = user_prompt_match.group(1).strip() if user_prompt_match else prompt

                    best_index = self.judge(image, plain_user_prompt, outputs_to_judge, type=stage_type)
                    winner = c1 if best_index == 0 else c2
                    judged_candidates.append(winner)
                    # Optional: Put loser back for different tournament styles, but simple removal works

                 # If candidates remain and we need more winners, add them (less ideal)
                 if temp_stage_candidates and len(judged_candidates) < beam_size:
                      judged_candidates.extend(temp_stage_candidates)

                 next_beams_input_ids = [cand['input_ids'] for cand in judged_candidates[:beam_size]]

            if not next_beams_input_ids:
                print(f"ERROR: No candidates survived judging for stage {stage}. Stopping generation.")
                # Again, need robust way to return previous best state
                return "Error: Generation failed, no candidates survived judging."

            current_beams = next_beams_input_ids # Update beams for the next stage
            print(f"  Selected {len(current_beams)} beams for the next stage.")
            # Optional: Print intermediate results for debugging
            # print(f"  Example beam text after {stage}: {self.processor.decode(current_beams[0][0][initial_length:], skip_special_tokens=True)[:150]}...")

        # After all stages, the best beam is the first one in current_beams (assuming some ranking)
        if not current_beams:
             return "Error: Generation finished with no final beams."
        final_output = self.processor.decode(current_beams[0][0][initial_length:], skip_special_tokens=True)
        print(f"--- Stage Beam Generation Finished ---")
        return final_output


    def generate(self, prompt, image_path, generation_type="stage", beam_size=2):
        """
        Public method to dispatch to the correct CoT generation strategy.

        Args:
            prompt (str): The user prompt (will be formatted with LLaVA template).
            image_path (str): Path to the input image file.
            generation_type (str): 'stage', 'sentence', or 'best_of_N'.
            beam_size (int): The beam size to use for the CoT methods.

        Returns:
            str: The generated text response.
        """
        if not self.model or not self.processor:
            return "Error: Model not loaded."
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        # Prepare the LLaVA-style prompt internally
        llava_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
        print(f"\n--- Starting Generation ---")
        print(f"Type: {generation_type}, Beam Size: {beam_size}")
        print(f"Image: {image_path}")
        print(f"Formatted Prompt: '{llava_prompt[:100]}...'")

        start_time = time.time()
        result = "Error: Invalid generation type specified."
        try:
            if generation_type == "stage":
                result = self.generate_inner_stage_beam(llava_prompt, image_path, beam_size)
            elif generation_type == "sentence":
                result = self.generate_inner_sentence_beam(llava_prompt, image_path, beam_size)
            elif generation_type == "best_of_N":
                result = self.generate_inner_best_of_N(llava_prompt, image_path, beam_size)
            else:
                 print(f"Error: Invalid generation_type '{generation_type}'")
                 raise ValueError("Invalid type. Choose from 'best_of_N', 'sentence', or 'stage'.")

            end_time = time.time()
            print(f"--- Generation Complete (Took {end_time - start_time:.2f} seconds) ---")
            return result

        except Exception as e:
             print(f"--- ERROR during generation ---")
             print(traceback.format_exc())
             return f"Error during generation: {e}"


    def measure_inference_time(self, prompt, image_path, generation_type, beam_size, num_runs=1, warmup_runs=0):
        """Measures wall and GPU time for a full CoT generation call."""
        print(f"\n--- Measuring Inference Time ({generation_type}, beam={beam_size}) ---")
        print(f"  (Running {warmup_runs} warmup + {num_runs} measurement runs)")
        wall_times = []
        gpu_times = []
        final_response_text = None

        if not os.path.exists(image_path):
            print(f"ERROR: Image path not found for timing: {image_path}")
            return -1.0, -1.0, "Error: Image not found"

        # Select the correct generation function based on type
        if generation_type == "stage":
            cot_function = self.generate_inner_stage_beam
        elif generation_type == "sentence":
            cot_function = self.generate_inner_sentence_beam
        elif generation_type == "best_of_N":
            cot_function = self.generate_inner_best_of_N
        else:
            print(f"ERROR: Invalid generation type '{generation_type}' for timing.")
            return -1.0, -1.0, f"Error: Invalid generation type {generation_type}"

        # Prepare the LLaVA-style prompt once
        llava_prompt = f"USER: <image>\n{prompt} ASSISTANT:"

        try:
            # Warm-up runs
            if warmup_runs > 0:
                 print("  Starting warm-up runs...")
                 for i in range(warmup_runs):
                     print(f"    Warm-up run {i+1}/{warmup_runs}...")
                     _ = cot_function(llava_prompt, image_path, beam_size=beam_size)
                     if self.device.type == 'cuda': torch.cuda.synchronize()
                 print("  Warm-up complete.")

            # Measurement runs
            print("  Starting measurement runs...")
            for i in range(num_runs):
                start_wall = time.perf_counter()
                start_gpu = None
                if self.device.type == 'cuda':
                    start_gpu = torch.cuda.Event(enable_timing=True)
                    end_gpu = torch.cuda.Event(enable_timing=True)
                    start_gpu.record()

                # --- Call the selected CoT generation function ---
                final_response_text = cot_function(llava_prompt, image_path, beam_size=beam_size)
                # ---------------------------------------------

                # Record end times
                if self.device.type == 'cuda':
                    end_gpu.record()
                    torch.cuda.synchronize() # Wait for GPU ops to finish
                end_wall = time.perf_counter()

                wall_times.append(end_wall - start_wall)
                if self.device.type == 'cuda':
                    gpu_times.append(start_gpu.elapsed_time(end_gpu) / 1000.0) # ms to s
                    print(f"  Run {i+1}/{num_runs}: Wall={wall_times[-1]:.4f}s, GPU={gpu_times[-1]:.4f}s")
                else:
                    gpu_times.append(0.0) # Append 0 for CPU runs
                    print(f"  Run {i+1}/{num_runs}: Wall={wall_times[-1]:.4f}s (GPU time not measured)")

            # Calculate averages
            avg_wall_time = sum(wall_times) / num_runs if wall_times else 0.0
            avg_gpu_time = sum(gpu_times) / num_runs if gpu_times and self.device.type == 'cuda' else 0.0

            print(f"  Average Inference Time: Wall={avg_wall_time:.4f}s, GPU={avg_gpu_time:.4f}s")
            if self.device.type != 'cuda': print("  (GPU time is 0.0 as device is CPU)")

        except Exception as e:
            print(f"  Error during CoT timing measurement: {e}")
            traceback.print_exc()
            avg_wall_time, avg_gpu_time, final_response_text = -1.0, -1.0, f"Error: {e}" # Indicate failure
        finally:
            print("----------------------------------------------------")
            # Return averages and the final text response from the last measured run
            return avg_wall_time, avg_gpu_time, final_response_text


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaVA-CoT Inference and Measure Metrics")

    # Model and Device Arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained LLaVA-CoT model directory.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for inference ('cuda' or 'cpu').")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Data type for model parameters ('bfloat16', 'float16', 'float32').")

    # Input Arguments
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image file.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="The text prompt to use with the image.")

    # CoT Generation Arguments
    parser.add_argument("--cot_type", type=str, default="stage", choices=["stage", "sentence", "best_of_N"],
                        help="Type of Chain-of-Thought generation strategy.")
    parser.add_argument("--beam_size", type=int, default=2,
                        help="Beam size for CoT generation methods.")

    # Measurement Arguments
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of measurement runs for timing inference.")
    parser.add_argument("--warmup_runs", type=int, default=0,
                        help="Number of warm-up runs before measurement.")
    parser.add_argument("--skip_flops", action="store_true",
                        help="Skip FLOPs estimation.")


    args = parser.parse_args()

    # --- Initialize Runner ---
    try:
        # Set default generation kwargs (can be customized further if needed)
        default_gen_kwargs = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)

        runner = LlavaCotRunner(
            model_path=args.model_path,
            device=args.device,
            dtype_str=args.dtype,
            generation_kwargs=default_gen_kwargs
        )
    except Exception as e:
        print(f"Failed to initialize LlavaCotRunner: {e}")
        exit(1)

    # --- Perform Measurements ---
    model_params, model_disk_gb = runner.get_model_size()

    estimated_gflops = None
    if not args.skip_flops:
        estimated_gflops = runner.estimate_flops()
    else:
        print("--- Skipping FLOPs estimation as requested ---")


    # --- Run Timed Inference ---
    avg_wall_time, avg_gpu_time, final_response = runner.measure_inference_time(
        prompt=args.prompt,
        image_path=args.image_path,
        generation_type=args.cot_type,
        beam_size=args.beam_size,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )

    # --- Print Final Summary ---
    print("\n" + "="*30 + " Final Summary " + "="*30)
    print(f"Model Path:          {args.model_path}")
    print(f"Device:              {runner.device} ({runner.dtype})")
    print("-" * 75)
    print(f"Image Path:          {args.image_path}")
    print(f"Prompt:              {args.prompt}")
    print(f"CoT Type:            {args.cot_type}")
    print(f"CoT Beam Size:       {args.beam_size}")
    print("-" * 75)
    print(f"Model Parameters:    {model_params/1e9:.2f} B" if model_params else "N/A")
    print(f"Model Disk Size:     {model_disk_gb:.2f} GB" if model_disk_gb is not None else "N/A")
    print(f"Est. Forward GFLOPs: {estimated_gflops:.2f}" if estimated_gflops is not None else ("N/A" if args.skip_flops else "Failed"))
    print("-" * 75)
    print(f"Avg Wall Time (s):   {avg_wall_time:.4f}" if avg_wall_time > 0 else "N/A (Timing Failed or Skipped)")
    print(f"Avg GPU Time (s):    {avg_gpu_time:.4f}" if avg_gpu_time > 0 and runner.device.type == 'cuda' else ("N/A (CPU or Timing Failed)" if runner.device.type == 'cuda' else "N/A (CPU)"))
    print(f"Timing Runs:         {args.num_runs} (Warmup: {args.warmup_runs})")
    print("-" * 75)
    print(f"Final Model Response:\n{final_response}")
    print("="*75)