import os
import json
import argparse
import traceback
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

class EvalRunner:
    def __init__(self, config_path, model_path_override=None):
        with open(config_path, 'r') as f:
            self.task_cfg_dict = json.load(f)

        # Optionally override model_path
        if model_path_override:
            self.task_cfg_dict['eval_config']['model'][0]['model_path'] = model_path_override

        self.work_dir = self.task_cfg_dict['eval_config'].get('work_dir', './outputs')
        os.makedirs(self.work_dir, exist_ok=True)

    def store_predictions_and_truth(self, predictions, ground_truth):
        output_file = os.path.join(self.work_dir, 'predictions_and_truth.json')
        data_to_store = [{"prediction": p, "ground_truth": g} for p, g in zip(predictions, ground_truth)]
        with open(output_file, 'w') as f:
            json.dump(data_to_store, f, indent=4)
        print(f"Predictions and ground truth saved to: {output_file}")

    def run(self):
        print("Running evaluation with config:")
        print(json.dumps(self.task_cfg_dict, indent=4))

        try:
            run_task(task_cfg=self.task_cfg_dict)
            print('>> Evaluation finished.')

            try:
                report_list = Summarizer.get_report_from_cfg(self.task_cfg_dict)
                print(f'\n>> Report List: {report_list}')
            except Exception as e:
                print(f"\n>> Error generating report: {e}")

            # Simulated report extraction
            predictions, ground_truth = [], []
            for report in report_list:
                predictions.append(report.get('prediction', 'N/A'))
                ground_truth.append(report.get('ground_truth', 'N/A'))

            self.store_predictions_and_truth(predictions, ground_truth)

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation from JSON config")
    parser.add_argument("--task_config_path", type=str, required=True, help="Path to the task config JSON file")
    parser.add_argument("--model_path", type=str, default=None, help="Override for merged model path")

    args = parser.parse_args()

    runner = EvalRunner(config_path=args.task_config_path, model_path_override=args.model_path)
    runner.run()
