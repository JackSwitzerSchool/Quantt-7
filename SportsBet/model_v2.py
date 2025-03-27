!pip install pandas scikit-learn requests

import requests
import pandas as pd
import io
from io import StringIO

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# -------------------------------------------------------------------
# DATA FETCHING FUNCTIONS
# -------------------------------------------------------------------

API_KEY = "cd3767d7233add126144319ddf41"  # your DataGolf API key

def fetch_player_list():
    """
    Get the list of players from DataGolf.
    """
    url = f"https://feeds.datagolf.com/get-player-list?file_format=csv&key={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    return df


def fetch_player_rankings():
    """
    Get DataGolf's player skill rankings.
    """
    url = f"https://feeds.datagolf.com/preds/get-dg-rankings?file_format=csv&key={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    rankings_df = pd.read_csv(StringIO(response.text))
    # Create a rank column if not provided
    if 'datagolf_rank' not in rankings_df.columns:
        rankings_df['datagolf_rank'] = rankings_df.index + 1
    return rankings_df


def fetch_pre_tournament_predictions(tour='pga'):
    """
    Get the pre-tournament predictions from DataGolf.
    This includes probabilities for win, top_5, top_10, top_20, etc.
    """
    url = f'https://feeds.datagolf.com/preds/pre-tournament?tour={tour}&file_format=csv&key={API_KEY}'
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    return df


def fetch_historical_results(start_date="2022-01-01", end_date="2023-01-01", tour="pga"):
    """
    Get historical tournament results from DataGolf for a given time range and tour.
    The endpoint can produce either CSV or JSON. We'll request CSV.
    
    NOTE: The actual DataGolf endpoint might differ; 
    consult the DataGolf docs for the correct / valid parameters.
    Below is an example that might work if DataGolf supports 
    'tournament-data/historical-results' or similar.

    We'll assume the columns we need are:
    - player_name
    - tournament_id
    - finish_position
    - etc.

    If the real endpoint is different, please adapt accordingly.
    """
    base_url = f"https://feeds.datagolf.com/historical-events?file_format=csv&key={API_KEY}"
    # Potentially pass in start/end dates as query params if supported
    # For example: &start_date=2022-01-01&end_date=2023-01-01&tour=pga
    url = f"{base_url}&start_date={start_date}&end_date={end_date}&tour={tour}"

    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    
    # Inspect or print columns if needed
    print("Historical columns:", df.columns)
    
    return df

# -------------------------------------------------------------------
# DATA MERGING / PREPARATION
# -------------------------------------------------------------------

def build_dataset():
    """
    Pull everything together: player list, rankings, pre-tournament predictions, 
    and historical results. Merge them into a single dataframe that includes
    the actual finishing position (to define top-10 outcome).
    """
    # 1) Player list and skill rankings
    player_list_df = fetch_player_list()
    rankings_df = fetch_player_rankings()

    # Merge these on 'player_name'
    # DataGolf typically uses 'player_name' consistently, 
    # but verify columns are consistent
    df_merged = pd.merge(
        player_list_df,
        rankings_df[['player_name', 'datagolf_rank']],
        how='left',
        on='player_name'
    )

    # 2) Pre-tournament predictions
    preds_df = fetch_pre_tournament_predictions(tour='pga')
    # We'll keep only columns needed: 'player_name','win','top_5','top_10','top_20'
    preds_keep_cols = ['player_name','win','top_5','top_10','top_20']
    for col in preds_keep_cols:
        if col not in preds_df.columns:
            print(f"Warning: column {col} not found in pre-tournament predictions.")
    preds_keep_cols = [col for col in preds_keep_cols if col in preds_df.columns]

    df_merged = pd.merge(
        df_merged,
        preds_df[preds_keep_cols],
        how='left',
        on='player_name'
    )

    # 3) Historical results to define top-10 outcome
    historical_df = fetch_historical_results(
        start_date="2022-01-01",
        end_date="2023-01-01",
        tour="pga"
    )
    # Data check:
    # print(historical_df.head())

    # We'll assume historical_df has columns: 
    # 'player_name', 'finish_position', 'tournament_id', 'event_name', 'start_date', etc.
    # Filter out players with missing finishing position
    historical_df = historical_df.dropna(subset=['finish_position'])
    
    # We'll define a binary variable top_10 = 1 if finish_position <= 10 else 0
    historical_df['top_10'] = (historical_df['finish_position'] <= 10).astype(int)

    # We want to join these historical results with the df_merged 
    # Possibly using 'player_name' and maybe 'event_name' or 'tournament_id'.
    # If we only want a single row per player, we might need to group by player_name 
    # to get some aggregated measure. 
    # But let's keep it at the event level for more robust data for modeling 
    # (each row is player-in-a-tournament).

    # Merge on 'player_name'. We'll create a final df that has one row per
    # (player_name, tournament_id). We'll also keep columns from df_merged 
    # for the player's skill rank, DataGolf predictions, etc.

    # However, df_merged currently might have 1 row per player, not per event.
    # We'll do a left merge from historical_df onto df_merged. 
    # This will replicate the player's skill/pred info across all tournaments they played in.
    final_df = pd.merge(
        historical_df,
        df_merged, 
        how='left',
        on='player_name'
    )

    # final_df now has columns from historical_df (tournament_id, finish_position, top_10, etc.)
    # plus columns from df_merged (datagolf_rank, predicted probs, etc.)

    return final_df

# -------------------------------------------------------------------
# MODELING
# -------------------------------------------------------------------
def train_model_on_top_10(final_df):
    """
    Train a Logistic Regression model to predict top-10 finishes
    based on DataGolf features and any other relevant columns.
    """

    # We'll pick a set of features. For instance:
    # - datagolf_rank
    # - win, top_5, top_10, top_20 from DataGolf (the 'pre_tournament predictions')
    # - You can add more, e.g., finishing_position from previous tournaments, etc.
    feature_cols = ['datagolf_rank','win','top_5','top_10','top_20']

    # Clean up feature columns, fill missing
    for col in feature_cols:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df[feature_cols] = final_df[feature_cols].fillna(0)

    # The target is 'top_10' from historical results
    # We'll remove rows with missing target
    model_df = final_df.dropna(subset=['top_10'])

    X = model_df[feature_cols]
    y = model_df['top_10'].astype(int)

    # Split train vs test 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a logistic regression (simple model)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC:  {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, feature_cols


# -------------------------------------------------------------------
# BETTING TABLE
# -------------------------------------------------------------------
def create_betting_table(final_df, model, feature_cols, edge_threshold=0.02):
    """
    For each player, compute model probability of top-10. Compare it
    to DataGolf's top_10 (the pre-tournament probability).
    Output a table with columns:
      - player_name
      - model_top_10_prob
      - dg_top_10_prob
      - edge (model - dg)
      - suggested bet (bool)
    We'll group by player_name, taking the *mean* of model probabilities if multiple tournaments are present.
    You might instead want the *latest* predictions for a given event.
    """

    # We'll focus on rows that have a valid top_10 from DataGolf's pre-tournament predictions
    # and that can be scored by our model
    df = final_df.dropna(subset=['top_10','player_name']).copy()

    # Prepare the feature matrix for these rows
    X = df[feature_cols].fillna(0)

    # Predict model probabilities
    model_probs = model.predict_proba(X)[:,1]
    df['model_top_10_prob'] = model_probs

    # DataGolf's predicted probability of top_10 (from pre-tournament preds)
    if 'top_10' in df.columns:  # careful not to conflict with the target 'top_10'
        # rename DataGolf's top_10 pred column to avoid confusion
        # if the historical 'top_10' was used as target
        # The pre-tournament top_10 prob might also be named something else.
        # Suppose the pre-tournament column is 'pred_top_10' to avoid confusion
        df.rename(columns={'top_10':'dg_top_10_prob'}, inplace=True)
    else:
        df['dg_top_10_prob'] = 0.0

    # Now we have for each row:
    # - dg_top_10_prob: the DataGolf (pre-tournament) top-10 probability
    # - model_top_10_prob: our model's top-10 probability

    # Group by player_name if we want a single row per player
    # Might want to pick the LATEST event only or the AVERAGE across events
    # For demonstration, let's do a simple average
    grouped = df.groupby('player_name', as_index=False).agg({
        'model_top_10_prob':'mean',
        'dg_top_10_prob':'mean'
    })

    # Compute edge
    grouped['edge'] = grouped['model_top_10_prob'] - grouped['dg_top_10_prob']

    # Suggest bet if edge > some threshold
    grouped['suggested_bet'] = grouped['edge'] > edge_threshold

    # Sort by edge descending
    grouped.sort_values('edge', ascending=False, inplace=True)

    # Format columns
    grouped.rename(columns={
        'player_name':'Player',
        'model_top_10_prob':'Model_Top10_Prob',
        'dg_top_10_prob':'DG_Top10_Prob',
        'edge':'Model_Edge',
        'suggested_bet':'Bet?'
    }, inplace=True)

    # Return the final table
    return grouped.reset_index(drop=True)

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------

def main():
    print("Building dataset...")
    final_df = build_dataset()
    print("Dataset built. Number of rows:", len(final_df))

    print("\nTraining model on top-10 finishes...")
    model, feature_cols = train_model_on_top_10(final_df)

    print("\nCreating betting table...")
    bet_table = create_betting_table(final_df, model, feature_cols, edge_threshold=0.02)

    print("\n=== Betting Table (Sample) ===")
    # Show top 20 rows
    print(bet_table.head(20).to_string(index=False))

    # Save to CSV
    bet_table.to_csv("betting_table_top10.csv", index=False)
    print("\nBetting table saved to betting_table_top10.csv")

if __name__ == "__main__":
    main()
