import requests
import pandas as pd
import io
from io import StringIO  # Import StringIO from the io module

# Function to fetch and process player list data
def fetch_player_list():
    url = "https://feeds.datagolf.com/get-player-list?file_format=csv&key=cd3767d7233add126144319ddf41"
    
    try:
        # Fetch data from the API (requesting CSV format)
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the CSV response into a DataFrame using StringIO
        player_list_df = pd.read_csv(io.StringIO(response.text))
        
        # Inspect the first few rows and column names
        print("Player List Columns:")
        print(player_list_df.columns)
        print(player_list_df.head())  # Print the first few rows of player list
        
        return player_list_df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player list: {e}")
        return None

# Function to fetch and process rankings data
def fetch_player_rankings():
    url = "https://feeds.datagolf.com/preds/get-dg-rankings?file_format=csv&key=cd3767d7233add126144319ddf41"
    
    try:
        # Fetch data from the API (requesting CSV format)
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the CSV response into a DataFrame using StringIO
        rankings_df = pd.read_csv(io.StringIO(response.text))
        
        # Inspect the first few rows and column names
        print("Player Rankings Columns:")
        print(rankings_df.columns)
        print(rankings_df.head())  # Print the first few rows of rankings
        
        # Add a column for the player's rank based on their position in the DataFrame
        rankings_df['rank'] = rankings_df.index + 1  # Rank starts from 1, so add 1 to index
        
        return rankings_df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player rankings: {e}")
        return None

# Function to merge player list and rankings
# Function to merge player list and rankings
def merge_player_list_with_rank():
    player_list = fetch_player_list()
    player_rankings = fetch_player_rankings()

    if player_list is not None and player_rankings is not None:
        # Merge player list with rankings on 'player_name' column (corrected column name)
        print("Merging player list with rankings...")
        merged_df = pd.merge(player_list, player_rankings[['player_name', 'datagolf_rank']], how='left', on='player_name')
        return merged_df
    else:
        print("Error: Player list or rankings data is missing.")
        return None

# Fetch and print the merged player list with rank
merged_data = merge_player_list_with_rank()

# Check if the data was successfully merged
if merged_data is not None:
    # Sort the merged data by the 'datagolf_rank' column
    merged_data_sorted = merged_data.sort_values(by='datagolf_rank', ascending=True)

    # Reset the index after sorting
    merged_data_sorted.reset_index(drop=True, inplace=True)

    # Display the sorted data
    print(merged_data_sorted[['player_name', 'datagolf_rank']].head())  # Display first few sorted players

else:
    print("No merged data available.")


def fetch_pre_tournament_predictions():
    url = 'https://feeds.datagolf.com/preds/pre-tournament?tour=pga&file_format=csv&key=cd3767d7233add126144319ddf41'
    response = requests.get(url)
    pre_tournament_df = pd.read_csv(StringIO(response.text))
    
    # Inspect the columns and show a few rows of the DataFrame
    print("Pre-tournament predictions columns:", pre_tournament_df.columns)
    print(pre_tournament_df.head())  # Show the first few rows to identify the correct column name
    
    return pre_tournament_df

pre_tournament_predictions = fetch_pre_tournament_predictions()

print("Pre-tournament predictions columns:", pre_tournament_predictions.columns.tolist())
print(pre_tournament_predictions.head())  # Show first few rows


# Step 2: Merge pre-tournament predictions with the merged data (which already has player list and rankings)
def merge_pre_tournament_predictions(merged_data):
    pre_tournament_predictions = fetch_pre_tournament_predictions()

    # Check the columns of the predictions DataFrame
    print("Merged data columns:", merged_data.columns)
    
    # Assuming the correct column name for predicted score is something like 'predicted_score'
    # Replace 'predicted_score' with the correct column name once we inspect the data
    merged_data_with_predictions = pd.merge(merged_data, pre_tournament_predictions[['player_name', 'top_20', 'top_10', 'top_5', 'win']], 
                                        how='left', on='player_name')

    return merged_data_with_predictions

# Step 3: Fetch rankings and player list, then merge them
merged_data = merge_player_list_with_rank()

# Add pre-tournament predictions
merged_data_with_predictions = merge_pre_tournament_predictions(merged_data)

# Step 4: Sort by rank
merged_data_sorted_with_predictions = merged_data_with_predictions.sort_values(by='datagolf_rank', ascending=True)

# Reset the index and display the results
merged_data_sorted_with_predictions.reset_index(drop=True, inplace=True)
print("Columns in merged_data_sorted_with_predictions:", merged_data_sorted_with_predictions.columns)

# Display the sorted and merged data with pre-tournament predictions
print(merged_data_sorted_with_predictions[['player_name', 'datagolf_rank']].head())
merged_data_sorted_with_predictions.to_csv("golf_predictions.csv", index=False)
print("CSV file saved as 'golf_predictions.csv'")