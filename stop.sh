#!/bin/bash
cd "$(dirname "$0")"

if [ -f proxy.pid ]; then
    PID=$(cat proxy.pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Proxy stopped (PID: $PID)"
    else
        echo "Process $PID not running"
    fi
    rm -f proxy.pid
else
    echo "No PID file found"
    # Try to find and kill anyway
    pkill -f "bedrock-proxy-go" 2>/dev/null && echo "Killed orphan proxy process" || true
fi
