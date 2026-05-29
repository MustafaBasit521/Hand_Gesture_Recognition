#!/bin/bash

# Activate virtual environment and run the Air-Draw app
source "$(dirname "$0")/venv/bin/activate"
streamlit run "$(dirname "$0")/app/air_draw_app.py"
