import requests
import pandas as pd
import io
from io import StringIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# 1. Data fetching
def fetch_player_list():
    url = "https://feeds.datagolf.com/get-player-list?file_format=csv&key=cd3767d7233add126144319ddf41"
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

def fetch_player_rankings():
    url = "https://feeds.datagolf.com/preds/get-dg-rankings?file_format=csv&key=cd3767d7233add126144319ddf41"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df['datagolf_rank'] = df.index + 1
    return df

def fetch_pre_tournament_predictions():
    url = 'https://feeds.datagolf.com/preds/pre-tournament?tour=pga&file_format=csv&key=cd3767d7233add126144319ddf41'
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def fetch_historical_results():
    """Simulated data. Replace with real historical results in practice."""
    data = {
        'player_name': ['Scottie Scheffler', 'Rory McIlroy', 'Jon Rahm', 'Justin Thomas', 'Tiger Woods',
                        'Scottie Scheffler', 'Rory McIlroy', 'Jon Rahm', 'Justin Thomas', 'Tiger Woods'],
        'tournament': ['Event A','Event A','Event A','Event A','Event A',
                       'Event B','Event B','Event B','Event B','Event B'],
        'finish_position': [1,3,5,25,50,2,8,10,22,55],
    }
    return pd.DataFrame(data)

def merge_player_list_with_rank():
    player_list = fetch_player_list()
    rankings = fetch_player_rankings()
    merged = pd.merge(
        player_list, 
        rankings[['player_name','datagolf_rank']], 
        how='left', 
        on='player_name'
    )
    return merged

def merge_all_data():
    df_merged = merge_player_list_with_rank()
    preds = fetch_pre_tournament_predictions()
    df_merged = pd.merge(
        df_merged, 
        preds[['player_name','top_20','top_10','top_5','win']],
        how='left', 
        on='player_name'
    )
    hist = fetch_historical_results()
    df_merged = pd.merge(
        df_merged, 
        hist, 
        how='left', 
        on='player_name'
    )
    return df_merged

def main():
    # 2. Merge everything
    full_data = merge_all_data()
    
    # 3. Create target variable: Did player finish top 20 historically?
    full_data['made_top_20'] = (full_data['finish_position'] <= 20).astype(int)
    
    # 4. Basic feature engineering
    numeric_cols = ['datagolf_rank','top_20','top_10','top_5','win']
    for col in numeric_cols:
        full_data[col] = pd.to_numeric(full_data[col], errors='coerce').fillna(0)
    
    # Example: average finishing position
    hist_agg = full_data.groupby('player_name')['finish_position'].mean().reset_index()
    hist_agg.rename(columns={'finish_position':'avg_finish_position'}, inplace=True)
    full_data = pd.merge(full_data, hist_agg, on='player_name', how='left')
    
    # 5. Prepare features & target
    feature_columns = ['datagolf_rank','top_20','top_10','top_5','win','avg_finish_position']
    target_column = 'made_top_20'
    
    model_df = full_data.dropna(subset=[target_column])
    X = model_df[feature_columns]
    y = model_df[target_column].astype(int)
    
    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    # 7. Model training (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 8. Evaluate
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:,1]
    print("\n=== Random Forest Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))
    
    # 9. Potential bets (example)
    # We'll compare the model's predicted probability for top 20 to DataGolf's top_20
    test_results = X_test.copy()
    test_results['player_name'] = model_df.loc[X_test.index, 'player_name']
    test_results['model_prob_top_20'] = rf_model.predict_proba(X_test)[:,1]
    test_results['dg_prob_top_20'] = test_results['top_20']
    
    # "Implied" prob from the DataGolf column (substitute real odds data if available)
    test_results['implied_prob'] = test_results['dg_prob_top_20']
    test_results['bet_value'] = test_results['model_prob_top_20'] - test_results['implied_prob']
    
    test_results.sort_values('bet_value', ascending=False, inplace=True)
    
    print("\n=== Potential Bets ===")
    print(test_results[['player_name','model_prob_top_20','implied_prob','bet_value']].head(10))

if __name__ == "__main__":
    main()
