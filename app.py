from flask import Flask, jsonify, render_template, request, send_from_directory, send_file
import os
import pandas as pd
import logging
import json
import re
import threading
import time
from io import BytesIO
from google.cloud import storage
import google.generativeai as genai
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import pyarrow  # Ensure it's installed
import fastparquet  # Ensure it's installed

# Configure Google AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__, template_folder="templates", static_folder="static")

# Caching Data
players_df = None
stats_df = None
player_ids = None

BUCKET = "mlb-eq-data"

def load_parquet_data():
    """Loads Parquet files asynchronously."""
    global players_df, stats_df, player_ids
    logging.info("Loading Parquet data...")

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET)

        logging.info(f"Loading players data")

        # Load players data
        players_blob = bucket.blob("data/all_mlb_players.parquet")
        players_df_temp = pd.read_parquet(pd.io.common.BytesIO(players_blob.download_as_bytes()))
        
        logging.info(f"Loading stats data")

        # Load stats data
        stats_blob = bucket.blob("data/all_mlb_stats.parquet")
        stats_df = pd.read_parquet(pd.io.common.BytesIO(stats_blob.download_as_bytes()))

        # Determine each player's most recent team and league
        last_team = (
            stats_df
            .sort_values(['year', 'sportId'], ascending=[False, True])
            .drop_duplicates(subset=['player_id'])
            .set_index('player_id')
        )

        # Map last team and league to players_df
        players_df_temp['last_team'] = players_df_temp['id'].map(last_team['team'].to_dict())
        players_df_temp['last_team_id'] = players_df_temp['id'].map(last_team['team_id'].to_dict())
        players_df_temp['last_league'] = players_df_temp['id'].map(last_team['sport_abbrev'].to_dict())
        players_df_temp['last_year'] = players_df_temp['id'].map(last_team['year'].to_dict())
        players_df = players_df_temp
        players_df_temp = None

        # Extract player ID to name mapping
        player_ids = {
            row['id']: f"{row['fullName']} - {row['last_team']} ({row['last_league']}, {row['last_year']})"
            for _, row in players_df.iterrows()
        }
        logging.info(f"Loaded {len(player_ids)} players into memory.")

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e

@app.route("/")
def home():
    """Render the search page."""
    return render_template("search.html")

@app.route("/player_list")
def player_list():
    """Returns a list of players for autocomplete."""
    global player_ids

    if player_ids is None:
        load_parquet_data()
        # return jsonify({"error": "Data not loaded"}), 500

    return jsonify(player_ids)

@app.route("/player/<int:player_id>")
def player_page(player_id):
    """Loads the player profile page."""
    if players_df is None or stats_df is None:
        return "Data not loaded", 500

    player_info = players_df.loc[players_df["id"] == player_id]
    if player_info.empty:
        return "Player not found", 404

    player_info = player_info.iloc[0]
    player_stats = stats_df[stats_df["player_id"] == player_id].to_dict(orient="records")

    return render_template(
        "player.html",
        player=player_info,
        stats=player_stats
    )

def create_scouting_report_prompt(player_id):
    """Generates a structured 20-80 scouting report prompt that outputs JSON."""

    player_row = players_df.loc[players_df["id"] == player_id]
    if player_row.empty:
        return None

    player = player_row.iloc[0]

    # Full list of possible icons
    baseball_traits = [
        "Power Hitter", "Contact Machine", "Disciplined Eye", "Clutch Performer",
        "Gap-to-Gap Hitter", "Opposite Field Master", "Home Run Specialist",
        "Fastball Hunter", "Breaking Ball Destroyer", "Bat Speed Freak",
        "High Contact Rate", "Pure Swing", "Late-Inning Threat",
        "First-Pitch Aggressor", "Situational Hitter", "Gold Glove Hands",
        "Cannon Arm", "Quick Reflexes", "Framing Artist", "Outfield Range",
        "Infield Maestro", "Smooth Double-Play Turns", "Diving Stop Specialist",
        "Strong Throwing Accuracy", "Fence Fearless", "Pickoff Master",
        "Relay Specialist", "Versatile Defender", "Bunt Defense Expert",
        "Game-Changer on Defense", "Flamethrower", "Wipeout Slider",
        "Pinpoint Command", "Clutch Pitcher", "Sinking Action",
        "Strikeout Artist", "Deception Master", "High Spin Rate",
        "Late-Life Fastball", "Changeup Specialist", "Crafty Lefty",
        "Bullpen Fireman", "Durable Workhorse", "Quick Pitcher",
        "Mound General", "Elite Speed", "Base-Stealing Threat",
        "Smart Baserunner", "Explosive First Step", "Aggressive Runner"
    ]

    prompt = f"""
    You are a professional baseball scout generating a **20-80 scouting report** for the following MLB prospect **as if evaluating them before they reached their peak** (e.g., as though they were 25 or younger).
    
    - **Name**: {player["fullName"]}
    - **Position**: {player["primary_position"]}
    - **Position Type**: {player["primary_position_type"]}
    - **Team**: {player["last_team"]}
    - **Age**: {player["currentAge"]}
    - **Height**: {player["height"]}
    - **Weight**: {player["weight"]}
    - **Bat Side**: {player["bat_side"]}
    - **Pitch Hand**: {player["pitch_hand"]}
    
    ### **Instructions**
    Generate a **structured JSON scouting report** following this format:
    
    ```json
    {{
        "scouting_summary": "[Concise summary of the player, including standout skills, weaknesses, and projection]",
        "projection": {{
            "best_case": "[Best-case MLB outcome—what type of player if development fully clicks]",
            "likely_outcome": "[Most realistic MLB projection based on current trajectory]",
            "risk_factor": "[Low/Moderate/High] - [Explain the risk in 1 sentence]"
        }},
        "scouting_grades": {{
            "hit": {{
                "grade": "[20-80 Grade] (For hitters only)",
                "comments": "[Assessment of contact ability, bat path, plate approach, strike zone awareness]"
            }},
            "power": {{
                "grade": "[20-80 Grade] (For hitters only)",
                "comments": "[Raw vs. game power, exit velocity, potential for 20+ HRs]"
            }},
            "run": {{
                "grade": "[20-80 Grade]",
                "comments": "[Speed, acceleration, stolen base instincts]"
            }},
            "arm": {{
                "grade": "[20-80 Grade]",
                "comments": "[Throwing strength, accuracy, mechanics]"
            }},
            "glove": {{
                "grade": "[20-80 Grade]",
                "comments": "[Defensive instincts, hands, range, positional flexibility]"
            }},
            "fastball": {{
                "grade": "[20-80 Grade] (For pitchers only)",
                "comments": "[Velocity, movement, command, ability to miss bats]"
            }},
            "breaking_ball": {{
                "grade": "[20-80 Grade] (For pitchers only)",
                "comments": "[Curveball/slider quality, swing-and-miss potential]"
            }},
            "changeup": {{
                "grade": "[20-80 Grade] (For pitchers only)",
                "comments": "[Deception, fade, ability to keep hitters off balance]"
            }},
            "command": {{
                "grade": "[20-80 Grade] (For pitchers only)",
                "comments": "[Control over the strike zone, walk rates, ability to locate pitches]"
            }}
        }},
        "strengths": [
            "[Brief but descriptive key strength #1 (7-12 words)]",
            "[Brief but descriptive key strength #2 (7-12 words)]",
            "[Brief but descriptive key strength #3 (7-12 words)]"
        ],
        "weaknesses": [
            "[Brief but descriptive key weakness #1 (7-12 words)]",
            "[Brief but descriptive key weakness #2 (7-12 words)]",
            "[Brief but descriptive key weakness #3 (7-12 words)]"
        ],
        "mlb_comparison": "[Player comparison(s) with reasoning—highlight which aspects match specific players.]"
        "final_grade": "[Overall 20-80 Grade—should reflect the player’s most defining tool]",
        "final_summary": "[One-sentence summary of the player's long-term projection]",
        "icons": [
            "[Select 3-5 traits that **best** define this player: {', '.join(baseball_traits)}]"
        ]
    }}
    ```

    ### **Important Notes**
    - **Do NOT include markdown or formatting markers** (e.g., `json`, `html`).
    - The **icons** section should highlight 3-5 of the **most defining** skills, not just general strengths.
    - In scouting_grades:
        - For "Pitcher" position type players: **Only** include the following tools: fastball, breaking ball, changeup, command, arm, glove, and run.
        - For "Two-Way Player" position type players: Include **all* tools: hit, power, run, arm, glove, fastball, breaking ball, changeup, command.
        - For all other position type players (i.e. hitters): **Only** include the following tools: hit, power, run, arm, glove.
    - Assume the evaluation is from before their full MLB career played out (e.g., before reaching their prime years).
    - Assign realistic 20-80 grades based on standard MLB grading scale:
      - 50 = MLB Average
      - 55 = Above Average
      - 60 = Plus
      - 70 = Elite
      - 80 = Generational Talent
    - **Concise but insightful**—avoid generic phrasing. 
    """

    return prompt

@app.route("/api/scouting_report/<int:player_id>")
def get_scouting_report(player_id):
    """Returns an AI-generated scouting report as JSON, caching in GCS."""

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(f"scouting_reports/{player_id}.json")

    # Check cache first
    if blob.exists():
        return jsonify(json.loads(blob.download_as_text()))
    
    # Generate a new report if not cached
    prompt = create_scouting_report_prompt(player_id)
    if prompt is None:
        return jsonify({"error": "Player not found"}), 404

    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
    response = model.generate_content(prompt)

    try:
        # Clean AI response: Remove ```json ... ``` markdown formatting
        cleaned_response = re.sub(r"```json\s*|\s*```", "", response.text).strip()

        # Convert AI response into JSON
        scouting_report = json.loads(cleaned_response)

        # Convert icons into proper URLs
        scouting_report["icons"] = [
            f"/icon/{icon.replace(' ', '_')}" for icon in scouting_report["icons"]
        ]

        # Save to cache
        blob.upload_from_string(json.dumps(scouting_report, indent=2), content_type="application/json")

        return jsonify(scouting_report)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid AI response format"}), 500

def generate_missing_icon(trait_name):
    """Generates and uploads a missing icon to Google Cloud using Vertex AI Image Generation."""
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    icon_blob = bucket.blob(f"icons/icon_{trait_name.replace(' ', '_')}.png")

    if icon_blob.exists():
        return  # Skip if already exists

    # Define the prompt for generating the icon
    prompt = f"Neon electric blue tubes in the shape of a baseball player who excels at '{trait_name}', with text at the bottom that says '{trait_name}'"

    PROJECT_ID = "mlb-eq"
    LOCATION = "us-west1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Load the Image Generation Model
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    # Generate the icon
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        language="en",
        aspect_ratio="1:1",  # Square for icons
    )

    if images and images[0]._image_bytes:
        # Upload image to GCS
        icon_blob.upload_from_string(images[0]._image_bytes, content_type="image/png")
        logging.info(f"✅ Generated and uploaded icon: {trait_name}")
    else:
        logging.error(f"❌ Failed to generate icon for: {trait_name}")

@app.route("/icon/<trait_name>")
def get_icon(trait_name):
    """Returns the icon image file directly or generates one if missing."""
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    trait_name = trait_name.replace('.png','').replace('icon_','')
    icon_blob = bucket.blob(f"icons/icon_{trait_name.replace(' ', '_')}.png")

    # Generate if the icon does not exist
    if not icon_blob.exists():
        generate_missing_icon(trait_name)

    # Fetch image bytes from GCS
    image_bytes = icon_blob.download_as_bytes()

    # Serve the image as a PNG response
    return send_file(BytesIO(image_bytes), mimetype="image/png")

if __name__ == "__main__":
    threading.Thread(target=load_parquet_data, daemon=True).start()
    logging.info("Starting Flask server on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
