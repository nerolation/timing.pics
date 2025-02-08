#!/bin/bash
cd ~/ethereum/timing_games_building
#python3 run_ethdata_pipeline.py
#python3 build.py

cd ./timing.pics
cp ../last_updated.txt ./
cp ../app.py ./
cp ../missed_slot_over_time_chart.pkl ./
cp ../time_in_slot_scatter_chart.pkl ./
cp ../gamer_bars.pkl ./
cp ../missed_slot_bars.pkl ./
cp ../gamer_advantage_lines.pkl ./
cp ../gamer_advantage_avg.pkl ./
cp ../missed_market_share_chart.pkl ./
cp ../missed_reorged_chart.pkl ./
cp ../missed_mevboost_chart.pkl ./
    
    
cp -r ../assets/ ./
cp ../requirements.txt ./
cp ../Procfile ./
cp ../runner_timingpics.sh ./

git add app.py 
git add last_updated.txt 
git add missed_slot_over_time_chart.pkl 
git add time_in_slot_scatter_chart.pkl 
git add gamer_bars.pkl 
git add missed_slot_bars.pkl 
git add gamer_advantage_lines.pkl 
git add gamer_advantage_avg.pkl 
git add missed_market_share_chart.pkl 
git add missed_reorged_chart.pkl
git add missed_mevboost_chart.pkl 
git add assets/
git add requirements.txt 
git add Procfile 
git add ./runner_timingpics.sh
git commit -m "update progress"
git push origin main
