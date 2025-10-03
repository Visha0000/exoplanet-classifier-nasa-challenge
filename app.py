import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import io

# Sample CSV for fallback
SAMPLE_CSV = """kepid,kepoi_name,kepler_name,koi_disposition,koi_pdisposition,koi_score,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec,koi_period,koi_period_err1,koi_period_err2,koi_time0bk,koi_time0bk_err1,koi_time0bk_err2,koi_impact,koi_impact_err1,koi_impact_err2,koi_duration,koi_duration_err1,koi_duration_err2,koi_depth,koi_depth_err1,koi_depth_err2,koi_prad,koi_prad_err1,koi_prad_err2,koi_teq,koi_teq_err1,koi_teq_err2,koi_insol,koi_insol_err1,koi_insol_err2,koi_model_snr,koi_tce_plnt_num,koi_tce_delivname,koi_steff,koi_steff_err1,koi_steff_err2,koi_slogg,koi_slogg_err1,koi_slogg_err2,koi_srad,koi_srad_err1,koi_srad_err2,ra,dec,koi_kepmag
10797460,K00752.01,Kepler-227 b,CONFIRMED,CANDIDATE,1.0000,0,0,0,0,9.488035570,2.7750000e-05,-2.7750000e-05,170.5387500,2.160000e-03,-2.160000e-03,0.1460,0.3180,-0.1460,2.95750,0.08190,-0.08190,6.1580e+02,1.950e+01,-1.950e+01,2.26,2.600e-01,-1.500e-01,793.0,,,93.59,29.45,-16.65,35.80,1,q1_q17_dr25_tce,5455.00,81.00,-81.00,4.467,0.064,-0.096,0.9270,0.1050,-0.0610,291.934230,48.141651,15.347
10797460,K00752.02,Kepler-227 c,CONFIRMED,CANDIDATE,0.9690,0,0,0,0,54.418382700,2.4790000e-04,-2.4790000e-04,162.5138400,3.520000e-03,-3.520000e-03,0.5860,0.0590,-0.4430,4.50700,0.11600,-0.11600,8.7480e+02,3.550e+01,-3.550e+01,2.83,3.200e-01,-1.900e-01,443.0,,,9.11,2.87,-1.62,25.80,2,q1_q17_dr25_tce,5455.00,81.00,-81.00,4.467,0.064,-0.096,0.9270,0.1050,-0.0610,291.934230,48.141651,15.347
10811496,K00753.01,,CANDIDATE,CANDIDATE,0.0000,0,0,0,0,19.899139950,1.4940000e-05,-1.4940000e-05,175.8502520,5.810000e-04,-5.810000e-04,0.9690,5.1260,-0.0770,1.78220,0.03410,-0.03410,1.0829e+04,1.710e+02,-1.710e+02,14.60,3.920e+00,-1.310e+00,638.0,,,39.30,31.04,-10.49,76.30,1,q1_q17_dr25_tce,5853.00,158.00,-176.00,4.544,0.044,-0.176,0.8680,0.2330,-0.0780,297.004820,48.134129,15.436
10848459,K00754.01,,FALSE POSITIVE,FALSE POSITIVE,0.0000,0,1,0,0,1.736952453,2.6300000e-07,-2.6300000e-07,170.3075650,1.150000e-04,-1.150000e-04,1.2760,0.1150,-0.0920,2.40641,0.00537,-0.00537,8.0792e+03,1.280e+01,-1.280e+01,33.46,8.500e+00,-2.830e+00,1395.0,,,891.96,668.95,-230.35,505.60,1,q1_q17_dr25_tce,5805.00,157.00,-174.00,4.564,0.053,-0.168,0.7910,0.2010,-0.0670,285.534610,48.285210,15.597
10854555,K00755.01,Kepler-664 b,CONFIRMED,CANDIDATE,1.0000,0,0,0,0,2.525591777,3.7610000e-06,-3.7610000e-06,171.5955500,1.130000e-03,-1.130000e-03,0.7010,0.2350,-0.4780,1.65450,0.04200,-0.04200,6.0330e+02,1.690e+01,-1.690e+01,2.75,8.800e-01,-3.500e-01,1406.0,,,926.16,874.33,-314.24,40.90,1,q1_q17_dr25_tce,6031.00,169.00,-211.00,4.438,0.070,-0.210,1.0460,0.3340,-0.1330,288.754880,48.226200,15.509
"""

# Step 1: Data Loading & Preprocessing Function
@st.cache_data
def load_and_preprocess_data(file_path_or_str):
    try:
        if isinstance(file_path_or_str, str) and file_path_or_str.startswith('http'):
            df = pd.read_csv(file_path_or_str, comment='#', sep=',', on_bad_lines='skip')
        elif isinstance(file_path_or_str, str):
            df = pd.read_csv(file_path_or_str, comment='#', sep=',', on_bad_lines='skip')
        else:
            df = pd.read_csv(io.StringIO(file_path_or_str), comment='#', sep=',', on_bad_lines='skip')
        
        feature_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_duration', 'koi_impact', 'koi_teq', 'koi_insol']
        available_cols = [col for col in feature_cols if col in df.columns]
        df = df[available_cols + ['koi_disposition']].dropna(subset=available_cols + ['koi_disposition'])
        
        if len(df) == 0:
            st.error("No valid data after preprocessing. Check CSV format.")
            st.stop()
        
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['koi_disposition'])
        
        X = df[available_cols].values
        y = df['label'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        return X_train, y_train, X_test, y_test, scaler, le, available_cols, df['koi_disposition'].value_counts()
    except Exception as e:
        st.error(f"Error loading CSV: {e}. Using sample data.")
        df = pd.read_csv(io.StringIO(SAMPLE_CSV), comment='#', sep=',', on_bad_lines='skip')
        feature_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_duration', 'koi_impact', 'koi_teq', 'koi_insol']
        available_cols = [col for col in feature_cols if col in df.columns]
        df = df[available_cols + ['koi_disposition']].dropna(subset=available_cols + ['koi_disposition'])
        
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['koi_disposition'])
        
        X = df[available_cols].values
        y = df['label'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        return X_train, y_train, X_test, y_test, scaler, le, available_cols, df['koi_disposition'].value_counts()

# Step 2: PyTorch MLP Model
class ExoplanetClassifier(nn.Module):
    def __init__(self, input_size):
        super(ExoplanetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(X_train, y_train, input_size, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExoplanetClassifier(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    return model.to('cpu'), device

# Step 3: Evaluation Function
def evaluate_model(model, X_test, y_test, le):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test, predicted.numpy())
        report = classification_report(y_test, predicted.numpy(), target_names=le.classes_)
        cm = confusion_matrix(y_test, predicted.numpy())
    return acc, report, cm

# Step 4: Confusion Matrix Plot
def plot_confusion_matrix(cm, le_classes):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=le_classes, yticklabels=le_classes,
           title='Confusion Matrix',
           ylabel='True label', xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return fig

# Step 5: Streamlit UI
def main():
    st.title("Exoplanet Classifier: NASA Space Apps Challenge")
    st.write("Upload Kepler/TESS CSV or enter features to classify as CONFIRMED, CANDIDATE, or FALSE POSITIVE.")
    
    # Load data
    uploaded_file = st.file_uploader("Upload CSV (uses GitHub CSV if none)", type='csv')
    if uploaded_file is not None:
        csv_content = uploaded_file.getvalue().decode()
        file_path = csv_content
    else:
        file_path = 'https://raw.githubusercontent.com/Visha0000/exoplanet-classifier-nasa-challenge/main/cumulative_2025.10.03_08.57.32.csv'
    
    if st.button("Load & Train Model"):
        with st.spinner("Preprocessing and training..."):
            try:
                X_train, y_train, X_test, y_test, scaler, le, feature_cols, class_dist = load_and_preprocess_data(file_path)
                input_size = len(feature_cols)
                model, _ = train_model(X_train, y_train, input_size)
                
                # Evaluate
                acc, report, cm = evaluate_model(model, X_test, y_test, le)
                st.metric("Test Accuracy", f"{acc:.2%}")
                st.text("Classification Report:\n" + report)
                
                # Plot Confusion Matrix
                fig = plot_confusion_matrix(cm, le.classes_)
                st.pyplot(fig)
                
                # Cache
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.le = le
                st.session_state.feature_cols = feature_cols
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.input_size = input_size
                st.session_state.class_dist = class_dist
                
                st.success("Model trained! Class distribution:\n" + class_dist.to_string())
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    # Prediction Section
    if 'model' in st.session_state:
        st.subheader("Predict New Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Manual Entry:")
            inputs = {}
            for col in st.session_state.feature_cols:
                inputs[col] = st.number_input(f"{col} (default: 0)", value=float(0.0), key=col)
            if st.button("Predict Single"):
                new_data = np.array(list(inputs.values())).reshape(1, -1)
                new_scaled = st.session_state.scaler.transform(new_data)
                with torch.no_grad():
                    output = st.session_state.model(torch.tensor(new_scaled, dtype=torch.float32))
                    _, pred = torch.max(output, 1)
                    prob = torch.softmax(output, dim=1).numpy()[0]
                pred_label = st.session_state.le.inverse_transform([pred.item()])[0]
                st.success(f"Prediction: **{pred_label}**")
                st.write("Probabilities:")
                for i, p in enumerate(prob):
                    st.write(f"- {st.session_state.le.classes_[i]}: {p:.2%}")
        
        with col2:
            new_file = st.file_uploader("Upload New Data CSV", key="new")
            if new_file and st.button("Predict Batch"):
                try:
                    new_csv = new_file.getvalue().decode()
                    new_df = pd.read_csv(io.StringIO(new_csv), comment='#', sep=',', on_bad_lines='skip')
                    available_cols = [col for col in st.session_state.feature_cols if col in new_df.columns]
                    new_X = new_df[available_cols].fillna(0).values
                    new_scaled = st.session_state.scaler.transform(new_X[:, :len(available_cols)])
                    with torch.no_grad():
                        outputs = st.session_state.model(torch.tensor(new_scaled, dtype=torch.float32))
                        _, preds = torch.max(outputs, 1)
                    new_df['predicted_disposition'] = st.session_state.le.inverse_transform(preds.numpy())
                    st.dataframe(new_df[['kepid', 'koi_disposition', 'predicted_disposition']].head() if 'kepid' in new_df else new_df.head())
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
    
    # Retrain Option
    if st.checkbox("Retrain with Hyperparams") and 'X_train' in st.session_state:
        epochs = st.slider("Epochs", 10, 100, 50)
        lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
        if st.button("Retrain"):
            with st.spinner("Retraining..."):
                model, _ = train_model(st.session_state.X_train, st.session_state.y_train, st.session_state.input_size, epochs, lr)
                st.session_state.model = model
                st.success("Model Retrained!")
                st.rerun()

if __name__ == "__main__":
    main()
