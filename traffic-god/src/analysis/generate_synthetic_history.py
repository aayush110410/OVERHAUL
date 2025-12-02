import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

def generate_synthetic_data():
    input_dir = Path("data/processed/tomtom")
    output_file = input_dir / "flows_synthetic_history.parquet"
    
    # Load existing data to get locations and static properties
    existing_files = list(input_dir.glob("flows_*.parquet"))
    if not existing_files:
        print("No existing data found to base synthetic data on.")
        return

    # Use the first file as a template for locations
    template_df = pd.read_parquet(existing_files[0])
    locations = template_df[['lat', 'lon', 'free_flow_speed', 'free_flow_travel_time', 'frc']].drop_duplicates()
    
    print(f"Generating history for {len(locations)} locations...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    # Generate timestamps every 15 minutes
    timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    synthetic_rows = []
    
    for _, loc in locations.iterrows():
        lat, lon = loc['lat'], loc['lon']
        ff_speed = loc['free_flow_speed']
        ff_time = loc['free_flow_travel_time']
        frc = loc['frc']
        
        for ts in timestamps:
            # Simulate traffic pattern:
            # Peak at 9 AM (hour 9) and 6 PM (hour 18)
            hour = ts.hour + ts.minute / 60.0
            
            # Morning peak
            morning_peak = np.exp(-((hour - 9)**2) / 2)
            # Evening peak
            evening_peak = np.exp(-((hour - 18)**2) / 2)
            
            congestion_factor = 0.3 * morning_peak + 0.4 * evening_peak + 0.1 * np.random.random()
            
            # Current speed drops as congestion rises
            current_speed = int(ff_speed * (1 - congestion_factor))
            current_speed = max(5, current_speed) # Minimum speed 5 km/h
            
            # Travel time increases as speed drops
            # time = distance / speed. 
            # ff_time = dist / ff_speed => dist = ff_time * ff_speed
            dist = ff_time * ff_speed
            current_travel_time = int(dist / current_speed)
            
            synthetic_rows.append({
                'timestamp': ts,
                'lat': lat,
                'lon': lon,
                'current_speed': current_speed,
                'free_flow_speed': ff_speed,
                'current_travel_time': current_travel_time,
                'free_flow_travel_time': ff_time,
                'confidence': 1.0,
                'road_closure': False,
                'frc': frc
            })
            
    synthetic_df = pd.DataFrame(synthetic_rows)
    
    # Ensure timestamp is timezone aware (UTC) to match existing data if possible, 
    # but the template might be mixed. Let's just ensure it's compatible.
    # The existing data had: 2025-11-24 06:30:17.009302+00:00
    if synthetic_df['timestamp'].dt.tz is None:
        synthetic_df['timestamp'] = synthetic_df['timestamp'].dt.tz_localize('UTC')
        
    synthetic_df.to_parquet(output_file, index=False)
    print(f"Generated {len(synthetic_df)} synthetic rows to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data()
