import streamlit as st
import pandas as pd
import torch
import multiprocessing
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to allow imports from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from parent directory
try:
    from newalpi_experiments_embedding import run_experiments
    from testclassifier.model import WeightedFocalLoss
    # For data loading in explainability
    from datasets.multioutput.newalpi import NewAlpi
except ImportError as e:
    st.error(f"Error importing modules from parent directory: {e}")
    st.stop()

# Import local model
from model_explainable import ExactOnlineLSTM

st.set_page_config(page_title="NewAlpi Experiment & Explainability", layout="wide")

st.title("🧪 NewAlpi Platform")

tabs = st.tabs(["🚀 Experiment Runner", "🧠 Explainability", "📉 Drift Analysis"])

# ==========================================
# SHARED STATE & DATA LOADING
# ==========================================
# Helper to get data and model if not present, shared by Tab 2 & 3
def ensure_data_loaded(n_samples=500):
    if 'model' not in st.session_state or len(st.session_state['X']) < n_samples:
        with st.spinner(f"Loading/Retraining on {n_samples} samples..."):
            # Load Data
            dataset_params = {
                'input_win': 1720, 
                'output_win': 480, 
                'delta': 0, 
                'sigma': 120, 
                'min_count': 0
            }
            stream = NewAlpi(machine=4, **dataset_params)
            stream.Y.columns = stream.Y.columns.astype(str)
            label_names = list(stream.Y.columns)
            
            # Get a batch of data
            X_batch = []
            Y_batch = []
            count = 0
            for x, y in stream:
                X_batch.append(x) # x is typically a dict {0: alarm_id, 1: alarm_id...} 
                # We need to convert x to list of values
                Y_batch.append(list(y.values()))
                count += 1
                if count >= n_samples:
                    break
            
            # Preprocess X to tensor
            X_list = [list(x.values()) for x in X_batch]
            max_len = max(len(x) for x in X_list)
            X_padded = np.zeros((len(X_list), max_len))
            for i, x in enumerate(X_list):
                X_padded[i, :len(x)] = x
            
            X_tensor = torch.tensor(X_padded, dtype=torch.long)
            Y_tensor = torch.tensor(Y_batch, dtype=torch.float)
            
            # Train Model
            output_dim = len(label_names)
            model = ExactOnlineLSTM(input_dim=10, output_dim=output_dim) 
            
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            criterion = torch.nn.BCEWithLogitsLoss()
            
            # Simple training loop
            model.train()
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = criterion(logits, Y_tensor)
            loss.backward()
            optimizer.step()
            
            # Store predictions for analysis
            model.eval()
            with torch.no_grad():
                final_logits = model(X_tensor)
                final_probs = torch.sigmoid(final_logits)
            
            # Save to session state
            st.session_state['model'] = model
            st.session_state['X'] = X_tensor
            st.session_state['Y'] = Y_tensor
            st.session_state['probs'] = final_probs
            st.session_state['label_names'] = label_names
            st.session_state['X_list'] = X_list 
            st.session_state['n_samples'] = n_samples
            
        st.success(f"Model trained on {n_samples} samples! Loss: {loss.item():.4f}")

# ==========================================
# TAB 2: EXPLAINABILITY (SALIENCY)
# ==========================================
with tabs[1]:
    st.header("🧠 Model Explainability (Saliency Maps)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Setup")
        n_samples = st.slider("Samples to Load", 100, 2000, 500, key="exp_samples")
        
        if st.button("Load Data & Train", key="btn_train_exp"):
            ensure_data_loaded(n_samples)

    with col2:
        st.subheader("2. Analyze Sample")
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_tensor = st.session_state['X']
            Y_tensor = st.session_state['Y']
            label_names = st.session_state['label_names']
            X_list = st.session_state['X_list']
            
            sample_idx = st.number_input("Select Sample Index", 0, len(X_tensor)-1, 0)
            
            # Get Prediction for single sample
            model.eval()
            x_sample = X_tensor[sample_idx].unsqueeze(0) # (1, seq_len)
            with torch.no_grad():
                logits = model(x_sample)
                probs = torch.sigmoid(logits)
            
            # Show True vs Pred
            true_labels = [label_names[i] for i, val in enumerate(Y_tensor[sample_idx]) if val == 1]
            pred_labels = [label_names[i] for i, val in enumerate(probs[0]) if val > 0.5]
            
            st.write(f"**True Labels:** {true_labels}")
            st.write(f"**Predicted Labels:** {pred_labels}")
            
            # Select Target Class for Saliency
            relevant_indices = [i for i, val in enumerate(Y_tensor[sample_idx]) if val == 1]
            if not relevant_indices:
                relevant_indices = [i for i, val in enumerate(probs[0]) if val > 0.5]
            if not relevant_indices:
                relevant_indices = [0]
                
            target_class_idx = st.selectbox(
                "Select Target Label to Explain", 
                relevant_indices,
                format_func=lambda i: f"{label_names[i]} (Idx: {i})"
            )
            
            # Calculate Saliency
            saliency = model.get_saliency_map(x_sample, target_class_idx) 
            saliency = saliency[0].detach().numpy()
            
            # Visualize
            st.write("### Sequence Importance Heatmap")
            
            raw_seq = X_list[sample_idx]
            seq_len = len(raw_seq)
            saliency_seq = saliency[:seq_len]
            
            # Matrix reshape
            items_per_row = 10
            rows = (seq_len + items_per_row - 1) // items_per_row
            
            fig, ax = plt.subplots(figsize=(10, rows * 0.5))
            
            padded_len = rows * items_per_row
            data_viz = np.nan * np.zeros(padded_len)
            data_viz[:seq_len] = saliency_seq
            data_matrix = data_viz.reshape(rows, items_per_row)
            
            annot_viz = np.full(padded_len, "", dtype=object)
            for i, val in enumerate(raw_seq):
                annot_viz[i] = str(val)
            annot_matrix = annot_viz.reshape(rows, items_per_row)
            
            sns.heatmap(data_matrix, annot=annot_matrix, fmt="", cmap="Reds", cbar=True, ax=ax, xticklabels=False, yticklabels=False)
            st.pyplot(fig)
            
        else:
            st.info("Train a model first to analyze samples.")

        # --- SHAP ANALYSIS ---
        st.divider()
        st.subheader("3. SHAP Analysis (Beta)")
        
        if 'model' in st.session_state and st.button("Run SHAP Analysis"):
            # Import SHAP within function to avoid startup lag if not used
            try:
                import shap
            except ImportError:
                st.error("SHAP not installed. `pip install shap`")
                st.stop()
                
            from model_explainable import ShapWrapper
            model = st.session_state['model']
            X_tensor = st.session_state['X']
            
            # 1. Setup Wrapper
            wrapper = ShapWrapper(model)
            
            # 2. Prepare Background Data (Embeddings)
            # Take random subsample
            bg_indices = np.random.choice(len(X_tensor), 50, replace=False)
            X_bg = X_tensor[bg_indices]
            
            # Need to get embeddings manually for background
            # Careful with padding_idx clipping if needed, but model handles it
            with torch.no_grad():
                # Clip input just like model forward
                X_bg_clamped = torch.clamp(X_bg, max=model.num_embeddings - 1)
                emb_bg = model.embedding(X_bg_clamped) # (50, seq_len, emb_dim)
            
            # 3. Prepare Target Sample (Embeddings)
            # Use same sample_idx from Saliency
            target_sample = X_tensor[sample_idx:sample_idx+1]
            with torch.no_grad():
                X_target_clamped = torch.clamp(target_sample, max=model.num_embeddings - 1)
                emb_target = model.embedding(X_target_clamped)
                
            # 4. Initialize Explainer
            with st.spinner("Computing SHAP values (GradientExplainer)..."):
                explainer = shap.GradientExplainer(wrapper, emb_bg)
                shap_values = explainer.shap_values(emb_target)
                
                # shap_values is a list of arrays (one per output class) if multi-output usually
                # OR if single output, just one array.
                # model head output is (batch, output_dim)
                # shap_values will be list of length output_dim
                
                # We want SHAP for the 'target_class_idx' selected above
                # If target_class_idx not defined (user didn't use Saliency section), default to 0
                if 'target_class_idx' not in locals():
                    target_class_idx = 0
                    
                shap_val_target = shap_values[target_class_idx] # (1, seq_len, emb_dim)
                
                # Sum across embedding dimension to get importance per token
                # (1, seq_len)
                shap_sum = np.sum(shap_val_target, axis=2)
                
            # 5. Visualize
            st.write(f"### SHAP Values for Class: {label_names[target_class_idx]}")
            
            # Bar chart
            import altair as alt # Use charts
            
            # Filter non-padding
            raw_seq = X_list[sample_idx]
            
            # Use safe length (min of raw sequence and computed SHAP width)
            shap_width = shap_sum.shape[1]
            viz_len = min(len(raw_seq), shap_width)
            
            shap_seq = shap_sum[0, :viz_len]
            raw_seq_viz = raw_seq[:viz_len]
            
            if len(raw_seq) != shap_width:
                 st.warning(f"Sequence length mismatch: Raw={len(raw_seq)}, SHAP={shap_width}. Truncated to {viz_len}.")
            
            chart_df = pd.DataFrame({
                "Alarm ID": [str(x) for x in raw_seq_viz],
                "SHAP Value": shap_seq,
                "Color": ["Positive" if x > 0 else "Negative" for x in shap_seq]
            })
            
            # Reset index to use as x-axis order
            chart_df = chart_df.reset_index().rename(columns={"index": "Position"})
            
            c = alt.Chart(chart_df).mark_bar().encode(
                x='Position',
                y='SHAP Value',
                color=alt.Color('Color', scale=alt.Scale(domain=['Positive', 'Negative'], range=['#2ecc71', '#e74c3c'])),
                tooltip=['Alarm ID', 'SHAP Value']
            ).properties(title="Feature Contribution (Positive=Supports Class, Negative=Opposes)")
            
            st.altair_chart(c, use_container_width=True)
            
            st.caption("SHAP values indicate how much each alarm in the sequence pushes the model towards predicting this class.")

# ==========================================
# TAB 3: DRIFT ANALYSIS
# ==========================================
with tabs[2]:
    st.header("📉 Drift Analysis")
    st.write("Analyze performance stability (Concept Drift) and feature distribution changes (Data Drift).")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        n_samples_drift = st.slider("Samples to Analyze", 200, 2000, 1000, key="drift_samples")
        if st.button("Analyze Drift", key="btn_drift"):
             ensure_data_loaded(n_samples_drift)
    
    if 'model' in st.session_state:
        Y_tensor = st.session_state['Y']
        probs = st.session_state['probs']
        X_list = st.session_state['X_list']
        n_total = len(Y_tensor)
        
        # --- 1. Performance & Confidence over Time ---
        st.subheader("1. Rolling Performance & Confidence (Concept Drift)")
        
        window_size = st.slider("Rolling Window Size", 10, 200, 50)
        
        # Calculate rough accuracy/F1 per sample (simplified as Exact Match here for speed, or partial accuracy)
        # Let's use Subset Accuracy (Exact Match)
        preds = (probs > 0.5).float()
        exact_matches = (preds == Y_tensor).all(dim=1).float().numpy()
        
        # Mean Confidence (avg of max probability per sample? or avg of probabilities for predicted classes?)
        # Let's take mean confidence of predicted classes
        # For multi-label: this is tricky. Let's take global mean probability mass for true class?
        # Simpler: Average max probability
        max_probs, _ = torch.max(probs, dim=1)
        max_probs = max_probs.detach().numpy()
        
        # Rolling calculations
        rolling_acc = pd.Series(exact_matches).rolling(window=window_size).mean()
        rolling_conf = pd.Series(max_probs).rolling(window=window_size).mean()
        
        chart_df = pd.DataFrame({
            "Sample Index": range(n_total),
            "Rolling Exact Match": rolling_acc,
            "Rolling Max Confidence": rolling_conf
        })
        
        st.line_chart(chart_df.set_index("Sample Index"))
        st.caption("A drop in Exact Match often indicates Concept Drift (model failing). A drop in Confidence might indicate inputs significantly different from training data.")

        # --- 2. Data Distribution Shift ---
        st.subheader("2. Data Distribution Shift (Feature Drift)")
        
        # Compare first N (Reference) vs last N (Current)
        ref_size = int(n_total * 0.2) # First 20%
        curr_size = int(n_total * 0.2) # Last 20%
        
        st.write(f"Comparing Reference (First {ref_size}) vs Current (Last {curr_size})")
        
        # Flatten lists to count alarms
        ref_alarms = [item for sublist in X_list[:ref_size] for item in sublist if item != 0] # 0 might be padding/background
        curr_alarms = [item for sublist in X_list[-curr_size:] for item in sublist if item != 0]
        
        from collections import Counter
        ref_counts = Counter(ref_alarms)
        curr_counts = Counter(curr_alarms)
        
        # Create DF
        all_keys = set(ref_counts.keys()) | set(curr_counts.keys())
        drift_data = []
        for k in all_keys:
            # Normalize?
            ref_freq = ref_counts[k] / len(ref_alarms) if ref_alarms else 0
            curr_freq = curr_counts[k] / len(curr_alarms) if curr_alarms else 0
            drift_data.append({
                "Alarm ID": str(k),
                "Ref Freq": ref_freq,
                "Curr Freq": curr_freq,
                "Diff": curr_freq - ref_freq,
                "Abs Diff": abs(curr_freq - ref_freq)
            })
            
        df_drift = pd.DataFrame(drift_data)
        
        # Top 10 Shifting Features
        top_shift = df_drift.sort_values(by="Abs Diff", ascending=False).head(15)
        
        # Melt for bar chart
        df_melt = top_shift.melt(id_vars=["Alarm ID", "Diff", "Abs Diff"], value_vars=["Ref Freq", "Curr Freq"], var_name="Window", value_name="Frequency")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Alarm ID", y="Frequency", hue="Window", ax=ax2)
        plt.title("Top Feature Shifts (Alarm ID Frequency)")
        st.pyplot(fig2)
        
        st.caption("Alarms with large differences in frequency between start and end of the stream indicate Data Drift.")
         
    else:
        st.info("Click 'Analyze Drift' to load data and compute metrics.")
