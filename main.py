import pandas as pd
import statsmodels.api as sm


if __name__ == '__main__':
    # load large scored dataset
    df = pd.read_csv('large_dataset_scored.csv')

    # classify certain countires 
    America = ['UnitedStates', 'Canada', 'Mexico']
    Europe = ['Netherlands', 'UnitedKingdom', 'Denmark',
              'Italy', 'Portugal', 'Serbia', 'Austria',
              'Switzerland', 'Germany', 'France', 'Slovenia',
              'CzechRepublic', 'Estonia', 'Poland', 'Turkey',
              'Greece', 'Spain', 'Macedonia']
    df['is_European'] = df['OriginCountry'].isin(Europe)
    to_keep = America + Europe
    df = df[df['OriginCountry'].isin(to_keep)]
    df = df.dropna(subset=['is_European', 'pred_label'])
    X = df[['is_European']].astype(int)
    y = df['pred_label']

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print(model.summary())
    print(sum(df['is_European']))


