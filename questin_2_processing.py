import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Example structure: paste your cleaned (positive-only) data here ---

data_air_canada = {
    'Year': [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    'Average_true': [18.60,18.75,25.93,19.58,19.66,14.17,7.41,5.88,4.13,13.33],
    'Average_pred': [39.15,44.17,50.26,34.92,32.48,24.17,22.22,14.71,8.72,17.33],
    'Change': [63.19,-28.97,37.80,33.97,34.28,89.92,-45.99,12.08,-13.36,-1.53],
    'Airline': ['Air_Canada']*10
}

data_american = {
    'Year': [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    'Average_true': [33.80,24.01,22.41,20.09,12.97,9.43,5.99,6.69,8.86,13.39],
    'Average_pred': [16.62,11.31,11.42,7.67,6.49,5.11,2.34,3.76,4.67,7.14],
    'Change': [467.44,-29.28,-8.55,2.91,-40.59,-16.97,-21.57,20.78,-28.94,8.62],
    'Airline': ['American_Airlines']*10
}

data_delta = {
    'Year': [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    'Average_true': [48.10,49.04,46.58,38.18,23.81,27.95,37.38,17.39,18.97,27.18],
    'Average_pred': [27.85,24.20,26.92,25.00,13.39,15.62,22.43,7.36,9.27,11.65],
    'Change': [73.56,-2.75,-8.24,10.22,-15.02,10.79,-31.78,-2.46,-15.76,22.86],
    'Airline': ['Delta_Airlines']*10
}

data_spirit = {
    'Year': [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    'Average_true': [22.34,21.11,20.58,16.19,13.53,11.09,15.91,5.86,6.63,3.54],
    'Average_pred': [4.74,5.73,7.51,4.52,3.66,5.97,8.52,4.65,2.06,1.57],
    'Change': [66.92,-48.17,40.69,-22.43,27.09,-30.21,-13.44,-0.81,-10.50,-15.62],
    'Airline': ['Spirit_Airlines']*10
}

# Combine all
df_all = pd.concat([
    pd.DataFrame(data_air_canada),
    pd.DataFrame(data_american),
    pd.DataFrame(data_delta),
    pd.DataFrame(data_spirit)
], ignore_index=True)

# --- Clean numeric columns ---
for col in ['Average_true','Average_pred','Change']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# --- Correlation per airline ---
print("\nCorrelation matrices by airline:\n")
for airline, subdf in df_all.groupby('Airline'):
    corr = subdf[['Average_true','Average_pred','Change']].corr()
    print(f"\n{airline}:\n", corr.round(3))

    # Plot each as its own heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Matrix – {airline}')
    plt.show()
