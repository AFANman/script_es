@echo off
call .venv\Scripts\activate
python -m python.capture.rank_crawler --out rank_data
