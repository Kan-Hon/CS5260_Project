from transformers import Seq2SeqTrainer
import torch
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Make sure to not truncate input_ids here:
        if self.args.predict_with_generate:
            # Pass the full input_ids from the dataset as prompt
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            
            # You can pass additional generation arguments if needed.
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                )
            
                # Optionally, compute loss if labels are provided
                loss = None
                if "labels" in inputs:
                    # Forward pass in eval mode (without generation)
                    outputs = model(**inputs)
                    loss = outputs.get("loss")
                
            return (loss, generated_tokens, inputs.get("labels"))
        else:
            # Fallback to default behavior if generation isn’t enabled.
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
