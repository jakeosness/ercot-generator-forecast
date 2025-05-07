import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, 
                                     MultiHeadAttention, Add, TimeDistributed, 
                                     Embedding, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load Data
df = pd.read_csv("sequence_model_train_data.csv")
df["GasLoadRatio"] = df["Natural Gas Price per Million BTU"] / (df["ForecastedLoadTotal"] + 1e-6)

# Extract Hour and Day of Week
df["SCED Time Stamp"] = pd.to_datetime(df["SCED Time Stamp"])
df["Hour"] = df["SCED Time Stamp"].dt.hour
df["DayOfWeek"] = df["SCED Time Stamp"].dt.dayofweek

# Drop unused columns
df.drop(columns=["Forecast_IssueTime", "Forecast_Target", "SCED Time Stamp"], inplace=True)

# Categorical Columns
embedding_col = "Resource Name"
onehot_cols = ["Resource Type", "QSE", "DME", "Resource Tech"]
target_cols = [f"Submitted TPO-MW{i}" for i in range(1, 9)] + [f"Submitted TPO-Price{i}" for i in range(1, 9)]
excluded_cols = ["Date", "SCED_DateTime"] + target_cols + [embedding_col] + onehot_cols
numeric_cols = [c for c in df.columns if c not in excluded_cols]

# Encode "Resource Name" with LabelEncoder for embedding
df[embedding_col] = df[embedding_col].astype(str)
embed_le = LabelEncoder()
df[embedding_col] = embed_le.fit_transform(df[embedding_col])
os.makedirs("Model_Info2", exist_ok=True)
joblib.dump(embed_le, "Model_Info2/label_encoder_Resource Name.pkl")
n_embed_tokens = df[embedding_col].nunique()

# One-hot encode other categoricals
onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
df_onehot = pd.DataFrame(onehot_enc.fit_transform(df[onehot_cols]))
joblib.dump(onehot_enc, "Model_Info2/onehot_encoder.pkl")

# Normalize numeric features
scaler = StandardScaler()
df_numeric = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
joblib.dump(scaler, "Model_Info2/feature_scaler.pkl")

# Combine all features
X_all = pd.concat([df[[embedding_col]].reset_index(drop=True), df_onehot, df_numeric.reset_index(drop=True)], axis=1)
df_targets = df[target_cols].copy().reset_index(drop=True)

# Build Sequences
sequence_length = 24
X_seq, y_seq = [], []
df["Date"] = df["Date"].astype(str)
X_all["group"] = df[embedding_col].astype(str) + "_" + df["Date"]
df_targets["group"] = X_all["group"]

for group in X_all["group"].unique():
    x_group = X_all[X_all["group"] == group].drop(columns=["group"])
    y_group = df_targets[df_targets["group"] == group].drop(columns=["group"])
    if len(x_group) == sequence_length:
        X_seq.append(x_group.values)
        y_seq.append(y_group.values)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Scale targets
target_scaler = StandardScaler()
y_seq_reshaped = y_seq.reshape(-1, y_seq.shape[-1])
y_seq_scaled = target_scaler.fit_transform(y_seq_reshaped).reshape(y_seq.shape)
joblib.dump(target_scaler, "Model_Info2/target_scaler.pkl")

# Train/val split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_scaled, test_size=0.2, shuffle=False)

# Build Transformer Model (with split MW/Price heads)
def build_model(input_shape, n_embed_tokens, embed_dim, output_dim=16):
    inp_all = Input(shape=input_shape)  # (24, features)
    inp_embed = Input(shape=(input_shape[0],), dtype='int32')  # (24,)

    emb = Embedding(input_dim=n_embed_tokens, output_dim=embed_dim)(inp_embed)
    x = Concatenate(axis=-1)([inp_all, emb])

    x = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    ffn = TimeDistributed(Dense(512, activation='relu'))(x)
    ffn = Dropout(0.3)(ffn)

    # Split MW and Price heads
    mw_out = TimeDistributed(Dense(8), name="MW_Head")(ffn)
    price_out = TimeDistributed(Dense(8), name="Price_Head")(ffn)
    out = Concatenate(name="Final_Output")([mw_out, price_out])

    model = Model(inputs=[inp_all, inp_embed], outputs=out)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

model = build_model(
    input_shape=X_train.shape[1:],
    n_embed_tokens=n_embed_tokens,
    embed_dim=16,
    output_dim=y_train.shape[-1]
)

# Train
model.fit(
    [X_train, X_train[:, :, 0].astype(int)], y_train,
    validation_data=([X_test, X_test[:, :, 0].astype(int)], y_test),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Save final model
model.save("Model_Info2/sequence_transformer_model.keras")
print("✅ Training complete with improved embedding, FFN, and split output heads.")
