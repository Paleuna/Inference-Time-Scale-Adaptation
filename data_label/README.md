# Manual accuracy review

This folder is the **working review stage** between raw inference output and the final model trajectories:

```text
results/<condition>/*.pkl
    → data_label/<condition>/*.pkl
    → review with data_labeler.ipynb
    → model_trajectory/<condition>/*.pkl   FINAL corrected model trajectories
```

The `results/` files contain model responses and provisional `is_correct` values. Move or copy one condition's pickle files into its matching `data_label/<condition>/` folder before reviewing them. Keep the same filenames so the corrected files can be matched to their original datasets. The export function writes corrected copies to `model_trajectory/`; the reviewed working files remain in `data_label/`.

## Review a condition

1. Open [`data_labeler.ipynb`](data_labeler.ipynb) in Colab and mount Google Drive.
2. In the viewer cell, set `PATH` to the condition you are reviewing, for example:

   ```python
   PATH = "/content/drive/MyDrive/inference-time-scale-adaptation/data_label/deepseek-r1"
   viewer = MismatchViewer(PATH)
   viewer.run()
   ```

3. Inspect the flagged responses. `True` means the answer is correct, `False` means it is wrong, and `Invalid` means there is no usable answer. The viewer saves each label back to the pickle file when you click a button. `Save & Exit` closes the viewer.
4. After reviewing, run the export cell for **that condition** to update `is_correct` and write the final files to `model_trajectory/<condition>/`.

The viewer treats `(NOT A) OR B`, `(NOT A) AND B`, `A OR (NOT B)`, and `A AND (NOT B)` as **order-sensitive** functions. For the other functions, it compares the unordered set of the two variables. It also flags some long or multi-variable final answers for manual inspection.

## Configuration before export

The final cell of this notebook currently refers to the older `infer_scale_reason` Drive folder. Replace its four hard-coded input/output pairs with one condition at a time:

```python
ROOT = "/content/drive/MyDrive/inference-time-scale-adaptation"
CONDITION = "deepseek-r1"  # change for each model condition
update_is_correct_from_label(
    f"{ROOT}/data_label/{CONDITION}",
    f"{ROOT}/model_trajectory/{CONDITION}",
)
```

Use the exact condition folder name, including capitalization. The repository currently uses lowercase `qwen3-instruct` and `qwen3-instruct-force`.

**Important:** the notebook's current `convert_label()` ignores `Invalid` instead of setting `is_correct=False`. Before exporting reviewed files, replace that function with:

```python
def convert_label(label):
    if isinstance(label, bool):
        return label
    if isinstance(label, str):
        value = label.strip().lower()
        if value == "true":
            return True
        if value in {"false", "invalid"}:
            return False
    return None
```

The viewer only offers a selected subset of responses for review, so inspect additional cases if your accuracy audit requires them.

Pickle files can execute code when loaded; use files from a trusted source.
