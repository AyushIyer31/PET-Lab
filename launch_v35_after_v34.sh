#!/bin/bash
# Wait for v34 (PID 72141) to finish, then launch v35
V34_PID=72141
LOG_DIR="/Users/admin/Documents/Los-Altos-Hacks"

echo "[$(date)] Watching for v34 (PID $V34_PID) to complete..."
while kill -0 $V34_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] v34 finished. Checking result..."
tail -20 "$LOG_DIR/training_v34.log"

echo "[$(date)] Launching v35..."
cd "$LOG_DIR"
nohup /Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python -u train_v35.py \
    > "$LOG_DIR/training_v35.log" 2>&1 &
V35_PID=$!
echo "[$(date)] v35 launched with PID $V35_PID"
echo $V35_PID > "$LOG_DIR/v35.pid"
