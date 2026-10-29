# Data Card

## Dataset

- **Planned dataset:** APTOS 2019 Blindness Detection.
- **Task:** Five-class diabetic retinopathy severity classification from retinal fundus images.
- **Labels:** 0 No DR; 1 Mild; 2 Moderate; 3 Severe; 4 Proliferative DR.
- **Expected use:** Educational portfolio experimentation and software demonstration only.
- **Data access:** Obtain through the official dataset/competition route and follow its current terms. Do not redistribute the dataset from this repository.

The execution blueprint reports 3,662 labelled training images and class counts of 1,805 / 370 / 999 / 193 / 295. Re-check these against the downloaded data before relying on them; this card does not independently validate the files.

## Splitting and leakage prevention

- Use stratified train/validation/test splits with a recorded random seed.
- Keep the test set untouched until model selection and threshold tuning are complete.
- Store split manifests and class counts for reproducibility.
- Check for duplicate or near-duplicate leakage if practical.
- Never mix test performance into training decisions.

## Known limitations and risks

- A single public dataset may not represent real-world devices, populations, image quality, or clinical settings.
- Severe and other minority classes may have small evaluation counts, making per-class metrics unstable.
- Dataset labels may contain noise or disagreement.
- A model can learn acquisition artefacts or spurious correlations.
- A quality gate based on simple image heuristics can accept non-retinal images or reject valid ones.
- Confidence values may be poorly calibrated.
- Grad-CAM visualizations can be unstable or misleading.
- No external or clinical validation is implied.

## Data handling

- Raw data remains local and is excluded from Git.
- Do not upload real patient images to this student demo.
- Use synthetic/demo patient IDs.
- Store only the hash of an uploaded image, not the image itself.
- Avoid placing identifying or sensitive information in logs, filenames, screenshots, or demo recordings.

## Evaluation reporting

Document the split seed, class counts, model configuration, preprocessing, package versions, and test metrics. Report uncertainty and per-class counts alongside metrics. Never state that the model is clinically accurate or safe based only on this benchmark.
