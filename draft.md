
speech draft of presentation (<= 3 min)

http://65.109.75.3:3000/

Using energy consumption data from September and October 2025, along with weather data from that same period, our team built an interactive map. It makes energy insights easy to see and understand. The goal is to help decision-makers figure out which buildings to invest in first.

The map covers several types of energy: electricity, gas, steam, cooling, and heat.

We used machine learning models to predict how much energy each building should use, based on things like temperature, past usage, and building age. Everything is measured per square foot so buildings can be compared fairly.

Each building gets a score based on the gap between what we predicted and what actually happened. A higher score means the building likely needs attention sooner. We also show how confident we are in each score.

There is a chatbot built into the tool. You can ask it questions about the data, get help using the map, or simply find out which buildings need the most attention. Let me demonstrate.

If you have additional data for these buildings, you can upload it directly. The system will run predictions on it and flag whether that building's energy use looks abnormal and might need a maintenance check.

In the machine learning model training part, we used xgboost on the data, but we found out the r^2 score of gas utility type is not ideal. Then specifically, we design a training system dedicated to gas data, using LSTM models. The performance gets significantly better.

---

## ML Model Slides Draft (~2 min)

**[Slide 1: Data Cleaning]**

Before building any models, we ran the raw meter data through a seven-step cleaning pipeline. We started with about 1.5 million rows. First, we dropped rows with missing join keys and removed utility types with no signal, like OIL28SEC. We also excluded three buildings that had no matching metadata. Then we applied hard caps to remove sensor faults — for example, any electricity reading above 10,000 was dropped. Short gaps of up to 8 intervals were imputed, and meters that were mostly or entirely empty were removed. This gave us a clean, reliable dataset to build on.

**[Slide 2: XGBoost Models]**

For our baseline models, we trained XGBoost across all five utility types: electricity, gas, heat, steam, and cooling. Each model uses 25 input features — weather variables like temperature and humidity, building metadata like floor area and age, temporal features like hour of day, plus engineered features such as lag values and rolling statistics. The target is energy per square foot, so buildings can be compared fairly. We used the same hyperparameters across all utilities: 1,000 max trees, depth of 7, learning rate of 0.05, with early stopping at 50 rounds.

**[Slide 3: XGBoost Validation]**

On the October test set, four out of five utilities achieved R-squared above 0.92. Cooling and steam were the highest at around 0.97. However, gas was the weakest at just 0.65 — the signal is noisier and harder for a tree-based model to capture. This gap motivated us to build a dedicated gas model.

**[Slide 4: LSTM Gas Model]**

So we designed a dual-branch LSTM specifically for gas. The temporal branch takes 28 features through a 3-layer LSTM with hidden size 256. A separate static MLP embeds building metadata into a 32-dimensional vector. These two branches are fused through a head network. The model has 1.39 million parameters and was trained for 33 minutes on a Tesla T4 GPU. The result: R-squared jumped from 0.65 to 0.97 — a massive improvement over XGBoost on gas data.

**[Slide 5: Gas Model Comparison]**

We actually tested four different architectures on the gas data: CNN, LSTM, Transformer, and XGBoost. The LSTM clearly won on accuracy with an R-squared of 0.97. The Transformer came second at 0.90. XGBoost was the fastest at under 2 minutes and the smallest at 151 kilobytes, but its accuracy was much lower. The takeaway: gas consumption has strong temporal dependencies that sequential models like LSTM can capture, which tree-based models simply miss.
