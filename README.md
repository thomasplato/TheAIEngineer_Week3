# Capstone Week 3 - Transformer Models Summary
I have made the Capstone Week 3 with extensive help from Claude.ai and Gemini.
I have spent a lot of time working to understand the core concepts - as the attention mechanism, masks, positional encoding, etc. The main part of my time was spent on understanding the mechanisms - not on code generation.

But I could not help challenging Claude.ai to make an implementation for volatility prediction (from SPY):
The point would be to reuse as many parts from Capstone Week 3 as possible. I used data from my Alpaca.markets subscription when running locally (using my keys) or publicly on a csv file hosted on GitHub. 

## Conclusion: 
Even using Claude.ai and Gemini I think it was very useful to tweak the parameters and devices: not so surprising very slow with CPU, faster with T4 GPU. Most of the loss progress was done after 2000 steps. With the Capstone Week 3 file I worked with 4 and 8 attention heads - and even for num_heads=1. Differences were noticeable but not dramatic.  

The transformer for volatility prediction (with num_heads=4) achieved RMSE of  7.673% vs EWMA's 3.596%. In the Mincer-Zarnowitz regression R² was 0.0003 vs 0.5158 for EWMA. That's hardly surprising. The transformer underperformed EWMA baselines, which highlights that architectural sophistication doesn't guarantee success on noisy (very few) financial data. With only ~1,245 days of SPY data and 128 test sequences, the model was not able to learn much. The most valuable insight was understanding how multi-head attention transformer models can learn content-based dependencies between inputs.

I have created two notebooks: 

## 1. CapstoneWeek3 Notebook: Character-Level Language Model

This notebook implements a decoder-only transformer for character-level language modeling.
The model uses scaled dot-product attention, multi-head attention, and transformer blocks with Pre-LN structure.
Training uses cross-entropy loss to predict the next character from a small text corpus.
Key components include: `scaled_dot_product_attention`, `MultiHeadAttention`, `TransformerBlock`, and `PositionalEncoding`.
The model generates text samples using temperature-controlled sampling after training on approx. 35,000 characters.

## 2. Vol_transformer_public Notebook: Volatility Forecasting Model

This notebook adapts the same transformer architecture for predicting 5-day forward realized volatility of SPY.
Input features include log returns, absolute returns, high-low range, volume changes, and historical volatility.
The model outputs a single regression value (MSE loss) instead of vocabulary probabilities.
Evaluation uses Mincer-Zarnowitz regression and compares against naive and EWMA baseline forecasts.
Data comes from Alpaca.markets API (or GitHub CSV fallback) with proper chronological train/val/test splits.

## 3. Comments on Using Transformers for Volatility Prediction

The attention mechanism enables in principle the model to identify which past days are most relevant for current volatility.
Unlike EWMA which use fixed decay weights, transformers can learn dynamic, content-based dependencies.
Volatility clustering and regime changes could benefit from the model's ability to attend to distant past events.
The architecture generalizes well because attention doesn't distinguish between text tokens and numerical features.
However, financial data is noisier and less structured than language, making patterns harder to learn reliably. As mentioned above, the model was not able to learn much with only ~1,245 days of SPY data and 128 test sequences.

## 4. Key Differences Between the Two Models

| Aspect | Language Model | Volatility Model |
|--------|----------------|------------------|
| **Input Layer** | `nn.Embedding(vocab_size, d_model)` - discrete tokens | `nn.Linear(num_features, d_model)` - continuous features |
| **Output Layer** | `nn.Linear(d_model, vocab_size)` - softmax over vocabulary | `nn.Linear(d_model, 1)` - single regression value |
| **Loss Function** | Cross-entropy (classification) | Mean Squared Error (regression) |
| **Evaluation** | Loss, bits-per-character, qualitative samples | RMSE, MAE, Mincer-Zarnowitz R² |
| **Data Split** | Random or sequential (order matters less for training) | Strictly chronological (no future data leakage) |
