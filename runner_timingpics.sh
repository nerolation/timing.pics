#!/bin/bash
cd ~/ethereum/timing_games
./bin/python3 run_ethdata_pipeline.py
./bin/python3 build.py

cd ./timing.pics
cp ../last_updated.txt ./
cp ../app.py ./
cp ../missed_slot_over_time_chart.pkl ./
cp ../time_in_slot_scatter_chart.pkl ./
cp -r ../assets/ ./
cp ../requirements.txt ./
cp ../Procfile ./
cp ../runner_timingpics.sh ./

git add app.py 
git add last_updated.txt 
git add missed_slot_over_time_chart.pkl 
git add time_in_slot_scatter_chart.pkl 
git add assets/
git add requirements.txt 
git add Procfile 
git add ./runner_timingpics.sh
git commit -m "update progress"
git push origin main