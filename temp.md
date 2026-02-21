claude --dangerously-skip-permissions

claude --resume --dangerously-skip-permissions

This is a hackathon competition team. this is the "anarchy" team.  include this information in the @README.md   


Create a markdown file to explain the structure of the data set.


do we have actual and expected energy consumption data?

how do we train this model? if we don't have the labels.


does huggingface package have regression models?

You can identify facilities whose energy use is unusually sensitive to temperature or shows consistently elevated baseline load—signals that may indicate inefficiency or opportunity.


what does load-signals mean here?

how can we train a model to determine whether a building a more sensitive to temperature changing?

tell me about the time series data in this dataset. should we use cnn or transformers to handle them? and what should we do when we have both time-series data and non time series data (age)?

Show me the structure of the neural network model. and where it is defined



Create a folder for the LSTM approach similar to this.

  Combining Time Series + Static Features (the key question)

  The standard approach is a hybrid architecture:

  Time Series (meter + weather)        Static Features (age,
  area, floors)
          │                                        │
     [LSTM]                            [MLP /
  Embedding]
          │                                        │
     temporal_embedding                     static_embedding
          │                                        │
          └──────────── concatenate ───────────────┘
                            │
                         [MLP head]
                            │
                      predicted energy

include a config file that inherits from the config file in the src folder in the lstm folder dedicated for this lstm approach. 


Create a folder for the LSTM approach similar to this.

  Combining Time Series + Static Features (the key question)

  The standard approach is a hybrid architecture:

  Time Series (meter + weather)        Static Features (age,
  area, floors)
          │                                        │
       [LSTM]                            [MLP /
  Embedding]
          │                                        │
     temporal_embedding                     static_embedding
          │                                        │
          └──────────── concatenate ───────────────┘
                            │
                         [MLP head]
                            │
                      predicted energy

include a config file that inherits from the config file in the src folder in the lstm folder dedicated for this lstm approach. 



because this dataset has time series data, explore the probability of using reinforcemnt leraning rather than static model like cnn.

  3. The time series nature doesn't imply RL

  Time series data ≠ reinforcement learning. The temporal
  structure means you should use sequence-aware supervised
  models, which is exactly what your 1D CNN with sliding
  windows already does. Other strong options:
  - LSTM / GRU — explicitly model temporal dependencies
  - Transformer — attention over the sequence captures
  long-range patterns
  - Temporal Fusion Transformer — state-of-the-art for
  multi-horizon time series

  These are all supervised, not RL.


  modify so that the folder shares a similar sturcture as this one.


  Huber regression model

---
    Per-epoch (during training):                 
  - metrics/val_rmse and metrics/val_mae — in real units (denormalized), not 
  just raw MSE loss                                                          
  - metrics/val_r2 — so you can watch R² climb over epochs
  - histograms/weights and histograms/gradients — diagnose
  vanishing/exploding gradients

  End of training (once):
  - figures/pred_vs_actual — scatter plot, logged as an image
  - figures/residual_distribution — histogram of residuals across all
  buildings
  - hparams — log hyperparams (lr, seq_length, conv_channels, etc.) tied to
  final metrics, so you can compare runs in TensorBoard's HParams tab


  also, can tensorboard present hyperparamters for this training instance it used as well?


  what input features are there for the model currently?

match the lstm tensorboard structure with the cnn ones. 


don't we have static feature inputs as well? what are the shapes of all these inputs?

so do we need to kill previous process before we can forward a new run?

Transcribe this image into a markdown file.