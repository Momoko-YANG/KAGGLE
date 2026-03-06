## CSIRO Biomass: 2-Stage Pipeline with Octa-Tile FiLM & Robust Cleaning

This notebook presents a robust 2-Stage Inference Pipeline designed for the CSIRO Image2Biomass Prediction challenge. Our approach tackles the problem of small data and noisy labels by combining metadata prediction with a specialized biomass regression model.

## key Features
1. Robust Image Cleaning Pipeline:
   - Automatically removes artifacts like laser pointer dots and date stamps using HSV masking and inpainting.
   - Applies CLAHE for lighting normalization and Safe Cropping to remove border noise.
   - This ensures the model focuses purely on the vegetation.

2. Phase 1: Metadata Prediction (Auxiliary Task):
   - Since test images lack metadata (Month, Height, NDVI, Species, State), we first predict these using a dedicated ConvNeXtV2-Base model.
   - We use a Split-View approach (processing left/right halves) to handle panoramic distortions.

3. Phase 2: Octa-Tile FiLM Biomass Regression:
   - Octa-Tile Architecture: The image is split into 8 local tiles plus 1 global view to capture fine-grained texture details (like clover leaves) while maintaining global context.
   - FiLM (Feature-wise Linear Modulation): The predicted metadata from Phase 1 is injected into the model using FiLM layers, dynamically modulating the visual features based on the season, location, and species probabilities.
   - TTA (Test Time Augmentation): Horizontal flipping is used to stabilize predictions.

4. Consistency Post-Processing:
   - Ensures that component weights (Green, Clover, Dead) sum up exactly to the predicted Total Biomass.
   - Aggressive filtering of low-confidence predictions to reduce noise.