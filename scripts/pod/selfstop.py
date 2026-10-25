#!/usr/bin/env python3
"""Backstop run by the pod start command after the hard limit: TERMINATE this pod (the laptop fetches results
periodically; an idle GPU must never keep billing)."""
import os
import urllib.request

pod = os.environ.get("RUNPOD_POD_ID")
for name in ("RUNPOD_API_KEY", "SELF_KEY"):
    key = os.environ.get(name)
    if not (pod and key):
        continue
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}", method="DELETE")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("User-Agent", "apart-collusion-sprint/0.1")
    try:
        urllib.request.urlopen(req, timeout=30)
        print(f"backstop: pod terminated via {name}")
        break
    except Exception as e:
        print(f"backstop: {name} failed: {type(e).__name__}")
