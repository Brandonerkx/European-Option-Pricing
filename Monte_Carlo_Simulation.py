#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes formula
def bsm_price(S0, E, r, T, sigma):
    d1 = (np.log(S0/E) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    eurocall_bsmprice = S0 * norm.cdf(d1) - E * np.exp(-r * T) * norm.cdf(d2)
    europut_bsmprice = E * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return eurocall_bsmprice, europut_bsmprice

# Simulation functions
def simulate_euler(S0, r, sigma, T, M, n): 
    dt = T / M 
    S = np.zeros((M + 1, n)) 
    S[0] = S0 
    for i in range(1, M + 1):  
        Z = np.random.standard_normal(n)
        S[i] = S[i - 1] * (1 + r * dt + sigma * np.sqrt(dt) * Z) 
    return S  

def simulate_milstein(S0, r, sigma, T, M, n): 
    dt = T / M
    S = np.zeros((M + 1, n)) 
    S[0] = S0 
    for i in range(1, M + 1):  
        Z = np.random.standard_normal(n) 
        S[i] = S[i-1] + (r * S[i-1] * dt) + (sigma * S[i-1] * np.sqrt(dt) * Z) + (0.5 * sigma**2 * S[i-1] * dt * (Z**2 - 1)) 
    return S

def simulate_runge_kutta(S0, r, sigma, T, M, n):
    dt = T / M
    S = np.zeros((M + 1, n))
    S[0] = S0
    for i in range(1, M + 1):
        Z = np.random.standard_normal(n)
        dW = np.sqrt(dt) * Z
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * dW
        S[i] = S[i-1] * np.exp(drift + diffusion)
    return S

# Option pricing using simulations
def calculate_price(simulate_func, S0, E, r, sigma, T, M, n):
    S_T = simulate_func(S0, r, sigma, T, M, n)[-1, :]  
    eurocall_payoff = np.maximum(S_T - E, 0.0)
    europut_payoff = np.maximum(E - S_T, 0.0)
    eurocall_price = np.exp(-r*T) * eurocall_payoff.mean()
    europut_price = np.exp(-r*T) * europut_payoff.mean()
    return eurocall_price, europut_price

# ---- STREAMLIT APP STARTS HERE ----

st.title("European Option Pricing")
st.write("Monte Carlo Simulation using Euler-Maruyama, Milstein, and Runge-Kutta Methods. Option prices computed by averaging payoffs and discounting under a risk neutral framework. Number of simulations and steps kept constant at 1000, option prices calculated using BSM as a means of comparing results from the 3 numerical methods.")

# Sliders for input variables
S0 = st.slider('Initial Stock Price (S₀)', min_value=20.0, max_value=1000.0, value=100.0, step=1.0)
E = st.slider('Strike Price (E)', min_value=20.0, max_value=1000.0, value=100.0, step=1.0)
T = st.slider('Time to Maturity (T in years)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
r = st.slider('Risk-Free Rate (r)', min_value=0.0, max_value=0.10, value=0.05, step=0.001)
sigma = st.slider('Volatility (σ)', min_value=0.0, max_value=0.50, value=0.20, step=0.001)

# Keep constant 
n = 1000
M = 1000

# Perform simulations
S_euler = simulate_euler(S0, r, sigma, T, M, n)
S_milstein = simulate_milstein(S0, r, sigma, T, M, n)
S_rungekutta = simulate_runge_kutta(S0, r, sigma, T, M, n)

# Calculate option prices
eurocall_euler, europut_euler = calculate_price(simulate_euler, S0, E, r, sigma, T, M, n)
eurocall_milstein, europut_milstein = calculate_price(simulate_milstein, S0, E, r, sigma, T, M, n)
eurocall_rungekutta, europut_rungekutta = calculate_price(simulate_runge_kutta, S0, E, r, sigma, T, M, n)
eurocall_bsmprice, europut_bsmprice = bsm_price(S0, E, r, T, sigma)

# Create dataframe for results
option_prices_df = pd.DataFrame({
    'Type': ['European Call', 'European Put'],
    'Black-Scholes': [eurocall_bsmprice, europut_bsmprice],
    'Euler': [eurocall_euler, europut_euler],
    'Milstein': [eurocall_milstein, europut_milstein],
    'Runge-Kutta': [eurocall_rungekutta, europut_rungekutta]
})

# Show the results
st.dataframe(option_prices_df)

# Plotting Euler paths
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(S_euler, alpha=0.6)
ax1.set_title("Simulated Stock Price Paths (Euler)")
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Stock Price")
st.pyplot(fig1)

# Plotting Milstein paths
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(S_milstein, alpha=0.6)
ax2.set_title("Simulated Stock Price Paths (Milstein)")
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Stock Price")
st.pyplot(fig2)

# Plotting Runge-Kutta paths
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(S_rungekutta, alpha=0.6)
ax3.set_title("Simulated Stock Price Paths (Runge-Kutta)")
ax3.set_xlabel("Time Steps")
ax3.set_ylabel("Stock Price")
st.pyplot(fig3)

