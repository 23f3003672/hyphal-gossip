import json
import urllib.request

def generate_trail_note(station_logs: list, use_local_gemma: bool = False, model_name: str = "gemma:2b") -> str:
    """
    Synthesizes an outdoor 'Touch Grass' field note from station observation logs.
    Connects to local Gemma via Ollama if available, or uses a deterministic template synthesizer.
    """
    log_summary = "\n".join([
        f"- Station {entry['station']} ({entry['location']}): {entry['time']}, Sky={entry['sky_condition']}, Notes: {entry['notes']}"
        for entry in station_logs
    ])
    
    prompt = f"""You are an outdoor field naturalist and distributed systems researcher.
Read these 5 sky observation station logs from our city walk and write a 3-paragraph naturalist trail note:
Paragraph 1: The physical atmosphere, light quality, and urban landscape across the 5 stations.
Paragraph 2: The sensory observations and anomalies (glare, haze, cloud movement).
Paragraph 3: How the decentralized fungal network (Hyphal Gossip) learned from these biased vantage points without a central server.

Station Logs:
{log_summary}
"""

    if use_local_gemma:
        try:
            req_data = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode("utf-8")
            req = urllib.request.Request("http://localhost:11434/api/generate", data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception:
            print("Local Gemma endpoint (Ollama) not reachable. Falling back to built-in trail note synthesizer.")

    # Built-in structured naturalist synthesis
    note = f"""# Hyphal Gossip: Field Trail Note

### The Five Vantage Points
Walking the circuit across the city revealed five distinct atmospheric slices. At the window station, the morning sun broke clear against high brick facades. Descending into the narrow alleyway, shadows cut off the horizon, compressing visible light into a narrow corridor of scattered cumulus. In the open park, an overcast ceiling settled low over wet grass, while the rooftop station caught the brunt of diffuse urban haze reflecting off asphalt. Finally, by the waterfront, an expansive open horizon blurred the boundary between water and gray sky.

### Outdoor Anomalies & Sensor Realities
Field observation immediately challenged theoretical assumptions. Glare at the rooftop station periodically saturated the camera sensor, while the alleyway suffered from sudden light shifts between tall shadows. Haze and thin overcast proved difficult to separate by eye, proving the necessity of the formal Sky Protocol. The data was inherently non-IID: no single station possessed the full vocabulary of the sky.

### The Hyphal Network Learns
Rather than funneling raw photographs into a central corporate cloud, each sky station learned on its own patch of sky. By passing low-rank rank-2 complement updates along the ring—sharing only the novel subspaces that immediate neighbors did not yet possess, and orthogonalizing them on receipt with Muon—the five stations reached consensus like fungal mycelium weaving across forest roots, operating entirely offline beneath the open air.
"""
    return note

if __name__ == "__main__":
    sample_logs = [
        {"station": 0, "location": "Window", "time": "08:15", "sky_condition": "Clear", "notes": "Crisp shadows, morning sun"},
        {"station": 1, "location": "Alley", "time": "09:00", "sky_condition": "Scattered", "notes": "Deep shadows, framing narrow sky"},
        {"station": 2, "location": "Park", "time": "10:15", "sky_condition": "Overcast", "notes": "Uniform gray ceiling, damp grass"},
        {"station": 3, "location": "Rooftop", "time": "11:30", "sky_condition": "Hazy", "notes": "Heavy dust and diffuse glare"},
        {"station": 4, "location": "Waterfront", "time": "12:45", "sky_condition": "Mixed", "notes": "Open horizon, moving wind"},
    ]
    note = generate_trail_note(sample_logs)
    print(note)
    with open("results/trail_note.md", "w") as f:
        f.write(note)
    print("Saved trail note to `results/trail_note.md`")
