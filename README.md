# Are More Tokens Rational? Inference-Time Scaling in Language Models as Adaptive Resource Rationality

**arXiv 2026** · Zhimin Hu, Riya Roshan, and Sashank Varma

[![Paper](https://img.shields.io/badge/Paper-arXiv-brightgreen)](https://arxiv.org/abs/2602.10329) [![Cite](https://img.shields.io/badge/Cite-BibTeX-blue)](#citation)

**TL;DR.** We use a Variable Attribution Task to ask whether language models adapt their reasoning to task complexity. Models identify the variables that determine trial outcomes under a specified logical function. As the number of candidate variables and trials changes, the paper reports a shift from brute-force search toward analytic elimination. Instruction-tuned models struggle more with XOR and XNOR, while reasoning models remain more robust.

## Experimental flow

```text
dataset/*.pkl
    └── run_experiment.ipynb or run_experiment_with_direct_answer.ipynb
          └── results/<condition>/*.pkl                 provisional model output
                └── copy to data_label/<condition>/*.pkl
                      └── manually review accuracy with data_labeler.ipynb
                            └── model_trajectory/<condition>/*.pkl       FINAL reviewed trajectories
                                  └── run_llm_as_judge.ipynb
                                        └── analyzed_trajectory/analyzed_<condition>/*.pkl
                                                                    FINAL strategy-labeled trajectories
```

`results/` and `data_label/` are working stages. **`model_trajectory/` and `analyzed_trajectory/` contain the final data products.** To inspect the provided results, you only need [`result_analysis.ipynb`](result_analysis.ipynb) and a final trajectory folder; no model or judge API calls are needed. Use an `analyzed_trajectory/` folder for all performance **and** strategy analyses. A `model_trajectory/` folder supports performance analysis, but does not contain the judge's strategy labels.

## Repository layout

```text
.
├── README.md
├── dataset/
│   ├── data_generation.ipynb             # Generate the task datasets
│   └── 2_vars_adaptive_sample_5_*.pkl    # One dataset per logical function
├── run_experiment.ipynb                  # Standard inference
├── run_experiment_with_direct_answer.ipynb
│                                       # Forced direct-answer inference
├── results/                              # Provisional inference output
├── data_label/
│   ├── README.md                         # Manual review instructions
│   ├── data_labeler.ipynb                # Review and export corrected accuracy
│   └── <condition>/*.pkl                 # Working copies for manual review
├── model_trajectory/
│   ├── data_labeler.ipynb                # Another copy of the review notebook
│   └── <condition>/*.pkl                 # Final reviewed trajectories
├── run_llm_as_judge.ipynb                # Reasoning-strategy classification
├── analyzed_trajectory/
│   └── analyzed_<condition>/*.pkl        # Final trajectories with strategy labels
├── result_analysis.ipynb                # Performance and strategy analysis
└── llm_judge_eval/
    ├── annotations_100.json             # Human-labeled examples
    └── llm_judge_eval.ipynb             # Evaluate the strategy judge
```

The existing model trajectory folders cover DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-force, Qwen3-instruct, Qwen3-instruct-force, and Qwen3-thinking. The existing analyzed folders cover DeepSeek-R1, Qwen3-instruct, and Qwen3-thinking. Each pickle file contains Python objects; load pickle files only from trusted sources.

## Info for analysis

1. Open [`result_analysis.ipynb`](result_analysis.ipynb) in Colab or Jupyter.
2. Run the import and helper-function cells.
3. Set `folder_path` to **one** final folder. For the full notebook, choose an analyzed folder, for example:

   ```python
   folder_path = "/content/drive/MyDrive/inference-time-scale-adaptation/analyzed_trajectory/analyzed_deepseek-r1"
   df = load_data(folder_path)
   ```

4. Run the visualization and statistical-analysis cells in order. Strategy comparisons filter to `elimination` and `permutation`; the fitted landscape plot is saved as `landscape.pdf` in the notebook's working directory.

The notebook's current active `folder_path` points to `model_trajectory/deepseek-r1`. Change it to an analyzed folder before running the strategy cells. The optional pickle-inspector cell at the end also contains a historical Drive path; update `FILE_PATH` or skip that cell.

### What is stored in each file?

- Dataset rows contain `full_prompt`, `inputs`, `outcomes`, `ground_truth`, `num_vars`, `num_trials`, `trials_delta`, and `difficulty_ratio`.
- Model trajectories add `llm_raw_response`, `parsed_prediction`, `is_correct`, `char_count`, and `model_name`. In `model_trajectory/`, `is_correct` can reflect manual review.
- Analyzed trajectories add `strategy_type` (`elimination`, `permutation`, or `none`) and `strategy_analysis`.

## Info for reproduction

### 1. Set up Colab and Drive

Place the whole repository at `/content/drive/MyDrive/inference-time-scale-adaptation/` after mounting Google Drive. If your folder has another name, update the absolute path variables in the notebooks. Install the notebook dependencies:

```python
%pip install together numpy pandas matplotlib seaborn scipy statsmodels scikit-learn tqdm ipywidgets
```

The inference and judge notebooks call the **Together API** and can incur usage charges. Their API-key assignments currently contain `xxx`; provide your own key at runtime through a private secret or environment variable, and do not commit a real key. The empty working directories may need to be created after cloning, because Git does not retain empty directories.

### 2. Generate or use datasets

Run [`dataset/data_generation.ipynb`](dataset/data_generation.ipynb), or use the supplied `dataset/*.pkl` files. The generator specifies 10 logical functions, candidate-variable counts `3, 4, 5, 6, 7, 8, 10, 12, 14, 16`, trial offsets `0`–`5` above the computed minimum, and five samples per setting. A sample may be skipped if an unambiguous instance is not found within the retry limit.

### 3. Run inference into `results/`

Use [`run_experiment.ipynb`](run_experiment.ipynb) for standard inference or [`run_experiment_with_direct_answer.ipynb`](run_experiment_with_direct_answer.ipynb) for a forced direct answer. Set `MODEL_NAME` and use a separate `OUTPUT_DIR` for each condition, such as `results/deepseek-r1/` or `results/deepseek-v3-force/`. Check `INPUT_DIR`, too. **Create `OUTPUT_DIR` before running the standard inference notebook:** it checkpoints before its directory-creation code.

For example, in the standard notebook's configuration cell, use:

```python
ROOT = "/content/drive/MyDrive/inference-time-scale-adaptation"
CONDITION = "deepseek-r1"  # change for each model condition
INPUT_DIR = os.path.join(ROOT, "dataset")
OUTPUT_DIR = os.path.join(ROOT, "results", CONDITION)
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

For the direct-answer notebook, use the same pattern with a condition such as `deepseek-v3-force` and the matching `MODEL_NAME`.

These files contain provisional `is_correct` values. The standard notebook compares variable sets; the direct-answer notebook compares ordered lists. Manual review in the next stage applies the function-specific rule and can correct these values.

### 4. Review accuracy and export final model trajectories

Copy the relevant `results/<condition>/*.pkl` files to `data_label/<condition>/`. Follow [`data_label/README.md`](data_label/README.md) to review flagged answers and export corrected files to `model_trajectory/<condition>/`. Keep condition names and capitalization consistent across folders, especially in Colab's case-sensitive environment.

### 5. Label strategies and export final analyzed trajectories

In [`run_llm_as_judge.ipynb`](run_llm_as_judge.ipynb), set `input_dir` to a **final** `model_trajectory/<condition>/` folder and `output_dir` to the corresponding `analyzed_trajectory/analyzed_<condition>/` folder. The notebook prefixes each output filename with `analyzed_`. Its optional inspection cell contains a separate hard-coded path; update it or skip it.

To evaluate the judge against human annotations, use [`llm_judge_eval/llm_judge_eval.ipynb`](llm_judge_eval/llm_judge_eval.ipynb). Its majority-vote save cell precedes the cell that creates `results_df`, and some `results_df_` plots and exports precede the single-pass evaluation cell. Run the producer cells before their dependent cells.

### Local Jupyter

Install `jupyterlab` alongside the packages above, skip the `google.colab` Drive-mount cells, and replace `/content/drive/...` paths with local paths. The notebooks do not share a central configuration file.

## Citation

```bibtex
@article{hu2026more,
  title={Are More Tokens Rational? Inference-Time Scaling in Language Models as Adaptive Resource Rationality},
  author={Hu, Zhimin and Roshan, Riya and Varma, Sashank},
  journal={arXiv preprint arXiv:2602.10329},
  year={2026}
}
```
