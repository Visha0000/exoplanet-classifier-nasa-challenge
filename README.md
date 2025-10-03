#A World Away: Hunting for Exoplanets with AI ( NASA Space Apps Challenge)
Classifies Kepler exoplanet data using a PyTorch MLP model.

## Team Information
- **Team Name**: Solo Flow
- **Member**: Vishalakshi (solo participant)

## Setup
1. Clone the repository: `git clone https://github.com/Visha0000/exoplanet-classifier-nasa-challenge`
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`
4. Deployed app: https://exoplanet-classifier-nasa-challengegit-erfeu6hhoooufmtn6lwgwn.streamlit.app/
   
## Data
- **CSV**: `cumulative_2025.10.03_08.57.32.csv` (available in the repository or at [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative))

## Features
- Train a model on Kepler/TESS CSV data.
- Predict single or batch exoplanet dispositions (CONFIRMED, CANDIDATE, FALSE POSITIVE).
- Retrain with custom hyperparameters.
- Visualize results with a confusion matrix.

## Demo
[Insert video link here, e.g., YouTube or Google Drive URL]

