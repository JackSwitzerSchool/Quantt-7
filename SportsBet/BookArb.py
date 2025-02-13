import requests
import numpy as np
import time
import pandas as pd
import asyncio
import aiohttp
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example bookmaker API endpoints (Replace with real ones)
BOOKMAKERS = {
    "Bookmaker_A": "https://api.bookmakerA.com/odds",
    "Bookmaker_B": "https://api.bookmakerB.com/odds"
}

# Global variables for the machine learning model
model = None
data = []

# Database setup
conn = sqlite3.connect("arbitrage.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS opportunities (
        match_id TEXT PRIMARY KEY,
        team1_odds_A REAL,
        team2_odds_A REAL,
        team1_odds_B REAL,
        team2_odds_B REAL,
        arbitrage INTEGER
    )
""")
conn.commit()

async def fetch_odds():
    """Fetch odds asynchronously from multiple bookmakers"""
    odds_data = {}
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_bookmaker_odds(session, book, url) for book, url in BOOKMAKERS.items()]
        results = await asyncio.gather(*tasks)
        for book, result in zip(BOOKMAKERS.keys(), results):
            odds_data[book] = result
    return odds_data

async def fetch_bookmaker_odds(session, book, url):
    try:
        async with session.get(url, timeout=5) as response:
            return await response.json()
    except Exception as e:
        print(f"Error fetching odds from {book}: {e}")
        return []

def find_arbitrage(odds_data):
    """Identify and store arbitrage opportunities"""
    for match in odds_data.get("Bookmaker_A", []):
        match_id = match["id"]
        team1_odds_A = match["team1_odds"]
        team2_odds_A = match["team2_odds"]
        
        match_B = next((m for m in odds_data.get("Bookmaker_B", []) if m["id"] == match_id), None)
        if match_B:
            team1_odds_B = match_B["team1_odds"]
            team2_odds_B = match_B["team2_odds"]
            
            prob1 = 1 / max(team1_odds_A, team1_odds_B)
            prob2 = 1 / max(team2_odds_A, team2_odds_B)
            
            arbitrage = 1 if prob1 + prob2 < 1 else 0
            cursor.execute("""
                INSERT OR REPLACE INTO opportunities (match_id, team1_odds_A, team2_odds_A, team1_odds_B, team2_odds_B, arbitrage)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (match_id, team1_odds_A, team2_odds_A, team1_odds_B, team2_odds_B, arbitrage))
            conn.commit()
            
            if arbitrage:
                print(f"Arbitrage Opportunity! Match ID: {match_id}")
                bankroll = 1000  
                stake1 = (bankroll * prob1) / (prob1 + prob2)
                stake2 = (bankroll * prob2) / (prob1 + prob2)
                print(f"Suggested Bets: ${stake1:.2f} on Team 1, ${stake2:.2f} on Team 2\n")

def train_model():
    """Train a machine learning model to predict arbitrage opportunities"""
    global model
    df = pd.read_sql("SELECT * FROM opportunities", conn)
    if len(df) < 100:
        print("Not enough data to train the model yet.")
        return
    
    X = df[['team1_odds_A', 'team2_odds_A', 'team1_odds_B', 'team2_odds_B']]
    y = df['arbitrage']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")

def predict_arbitrage(odds_data):
    """Predict arbitrage opportunities using the trained model"""
    if model is None:
        print("Model not trained yet.")
        return
    
    for match in odds_data.get("Bookmaker_A", []):
        match_id = match["id"]
        team1_odds_A = match["team1_odds"]
        team2_odds_A = match["team2_odds"]
        
        match_B = next((m for m in odds_data.get("Bookmaker_B", []) if m["id"] == match_id), None)
        if match_B:
            team1_odds_B = match_B["team1_odds"]
            team2_odds_B = match_B["team2_odds"]
            
            features = np.array([[team1_odds_A, team2_odds_A, team1_odds_B, team2_odds_B]])
            prediction = model.predict(features)
            
            if prediction == 1:
                print(f"Predicted Arbitrage Opportunity! Match ID: {match_id}")

async def main():
    while True:
        odds_data = await fetch_odds()
        find_arbitrage(odds_data)
        train_model()
        predict_arbitrage(odds_data)
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
