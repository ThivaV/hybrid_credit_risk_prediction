# Hybrid Credit Risk Prediction with Reinforcement Learning and LSTM

* [REPO: Hybrid Credit Risk Prediction with Reinforcement Learning and LSTM](https://github.com/ThivaV/hybrid_credit_risk_prediction)
* [NOTEBOOK: Hybrid Credit Risk Prediction with Reinforcement Learning and LSTM](hybrid_credit_risk_prediction_v2.0.ipynb)

## Project Overview

The **Hybrid Credit Risk Prediction** project combines traditional machine learning (LSTM) and advanced reinforcement learning techniques to predict credit risk and make loan approval decisions. 
The model uses an LSTM network to classify borrower behavior based on past data, while a reinforcement learning agent optimizes decision-making by interacting with a custom environment where it either approves or rejects loans based on these predictions.

## Key Components

1. **LSTM Model:** Used for time-series prediction of credit risk based on historical features of borrowers.
2. **Reinforcement Learning Agent (DQN):** Learns an optimal loan approval strategy by interacting with a custom environment that simulates loan decisions.
3. **Custom Environment:** Defined using the gym framework, where the agent takes actions (approve/reject loans) based on the borrower data and receives rewards based on the correctness of its actions.

## Environment Setup

### Requirements

* Python 3.11
* Required libraries:
    * `scikit-learn`
    * `tensorflow`
    * `gym`
    * `stable-baselines3[extra]`
    * `pandas`
    * `numpy`
    * `matplotlib`

## Dataset Preparation

* [Dataset: GiveMeSomeCredit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)

* The dataset consists of historical borrower data, including financial features and a target label indicating whether the borrower defaulted (1) or repaid the loan (0).

    1. **Feature Scaling:** Standardize the features using StandardScaler to ensure better performance with both LSTM and the RL agent.
    2. **Training and Testing Split:** Split the data into X_train, X_test, y_train, and y_test for training and evaluation purposes.

## LSTM Model for Credit Risk Prediction

### LSTM Architecture

The **Long Short-Term Memory (LSTM)** model is used to predict whether a loan will be defaulted or not. This model handles the time-series nature of the problem effectively by considering sequential dependencies in the data.

### Evaluation

After training the LSTM model, it predicts whether the borrower will default or repay a loan, which will serve as the feedback for the RL agent during its decision-making process.

## Reinforcement Learning (DQN) for Loan Decision Making

### Custom Gym Environment

A custom environment (`CreditRiskEnv`) is defined to simulate loan approval decisions. The RL agent interacts with this environment to approve or reject loans, receiving rewards based on the LSTM model’s prediction.

### Environment Design

1. **Action Space:** The agent can take two actions:
    * `0`: Reject the loan.
    * `1`: Approve the loan.
2. **Observation Space:** The agent observes the borrower’s financial features (scaled) to make a decision.
3. **Rewards:**
    * A reward of `+1` is given for a correct loan approval (LSTM predicts repayment).
    * A penalty of `-1` is given for an incorrect loan approval (LSTM predicts default).
4. **Termination:** The episode ends when all data points have been used.

### Training the RL Agent

The RL agent is trained using **Deep Q-Networks (DQN)** from the `stable-baselines3` library. The DQN agent interacts with the environment and learns to make optimal loan approval decisions.

### Evaluation

After training, the DQN agent can be evaluated by resetting the environment and observing its performance in terms of rewards gained over multiple episodes.

## Conclusion

This project presents a hybrid approach to **credit risk prediction** by combining **LSTM-based predictions with Reinforcement Learning** decision making. The LSTM model forecasts borrower behavior, and the RL agent learns an optimal loan approval strategy based on these forecasts.

## Key Benefits

* **Sequential Learning:** LSTM handles the time-series aspect of borrower features.
* **Optimized Decision-Making:** The DQN agent optimizes loan approval based on real-time interactions with the environment.
* **Customizable:** The environment can be adapted to include more complex decision-making processes.

## Future Work

Possible extensions of this project include:
* Introducing more complex reward functions based on financial metrics (e.g., profit, risk).
* Expanding the observation space to include more borrower attributes.
* Exploring other RL algorithms like **PPO** or **A2C** for improved performance.

## References

* [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)
* [LSTM Networks for Time Series Prediction](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
* [Reinforcement Learning in Finance](https://www.coursera.org/learn/reinforcement-learning-in-finance?msockid=0afac952da2b64400c9fdd02db876505)
