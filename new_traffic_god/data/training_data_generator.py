"""
Training Data Generator for Traffic God
========================================

Generates training data for the custom LLM from:
1. Synthetic conversations about traffic
2. Traffic patterns and rules
3. Route planning scenarios
4. Q&A pairs about Noida/NCR traffic

This data is used to train YOUR OWN model - no external APIs.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import os


class TrafficDataGenerator:
    """
    Generates training data for the Traffic God LLM
    """
    
    # Noida locations
    SECTORS = [f"Sector {i}" for i in range(1, 169)]
    
    MAJOR_LOCATIONS = [
        "Noida City Centre", "Sector 18 Market", "DLF Mall",
        "Botanical Garden", "Golf Course", "Film City",
        "Sector 62", "Sector 63", "Tech Park",
        "Noida Expressway", "DND Flyway", "NH-24",
        "Indirapuram", "Vaishali", "Kaushambi",
        "Greater Noida", "Pari Chowk", "Knowledge Park"
    ]
    
    METRO_STATIONS = [
        "Noida City Centre", "Golf Course", "Botanical Garden",
        "Noida Sector 18", "Noida Sector 16", "Noida Sector 15",
        "Noida Sector 62", "Noida Sector 59", "Noida Sector 61",
        "Vaishali", "Kaushambi", "Anand Vihar"
    ]
    
    ROAD_TYPES = ["expressway", "main road", "sector road", "service road"]
    
    TRAFFIC_LEVELS = ["very light", "light", "moderate", "heavy", "very heavy", "gridlocked"]
    
    AQI_CATEGORIES = ["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.conversations = []
        self.qa_pairs = []
    
    def generate_traffic_query_response(self) -> Dict[str, str]:
        """Generate a traffic status Q&A pair"""
        location = random.choice(self.MAJOR_LOCATIONS + self.SECTORS[:50])
        hour = random.randint(0, 23)
        
        # Determine traffic based on time
        if 8 <= hour < 11:
            level = random.choice(["heavy", "very heavy"])
            speed = random.randint(10, 25)
            delay = random.randint(15, 40)
            period = "morning peak"
        elif 17 <= hour < 21:
            level = random.choice(["heavy", "very heavy", "gridlocked"])
            speed = random.randint(8, 20)
            delay = random.randint(20, 50)
            period = "evening peak"
        elif 11 <= hour < 17:
            level = random.choice(["moderate", "light"])
            speed = random.randint(30, 45)
            delay = random.randint(0, 10)
            period = "off-peak"
        else:
            level = random.choice(["very light", "light"])
            speed = random.randint(45, 60)
            delay = 0
            period = "night"
        
        # Question variations
        questions = [
            f"What's the traffic like at {location}?",
            f"How is traffic at {location} right now?",
            f"Is there congestion at {location}?",
            f"Traffic status for {location}",
            f"How busy is {location} currently?",
            f"What's the traffic situation near {location}?",
            f"Can you tell me about traffic at {location}?",
        ]
        
        # Response templates
        responses = [
            f"Traffic at {location} is currently {level}. Average speed is around {speed} km/h with an estimated delay of {delay} minutes. This is typical for {period} hours.",
            
            f"The traffic situation at {location} shows {level} congestion. Vehicles are moving at approximately {speed} km/h. {'Consider alternative routes or metro.' if level in ['heavy', 'very heavy', 'gridlocked'] else 'Good time to travel.'}",
            
            f"Current conditions at {location}: {level.title()} traffic with {speed} km/h average speed. {f'Expect delays of around {delay} minutes.' if delay > 0 else 'No significant delays.'}",
        ]
        
        return {
            "instruction": random.choice(questions),
            "response": random.choice(responses),
            "metadata": {
                "type": "traffic_query",
                "location": location,
                "hour": hour,
                "level": level
            }
        }
    
    def generate_route_query_response(self) -> Dict[str, str]:
        """Generate a route planning Q&A pair"""
        source = random.choice(self.MAJOR_LOCATIONS)
        destination = random.choice([l for l in self.MAJOR_LOCATIONS if l != source])
        
        distance = random.randint(5, 25)
        base_time = int(distance * 2)  # 2 min/km baseline
        
        hour = random.randint(0, 23)
        if 8 <= hour < 11 or 17 <= hour < 21:
            time_factor = random.uniform(1.8, 2.5)
            traffic_note = "Due to peak hour traffic, expect delays."
        else:
            time_factor = random.uniform(1.0, 1.3)
            traffic_note = "Traffic is relatively clear."
        
        estimated_time = int(base_time * time_factor)
        
        # Generate route options
        routes = [
            {"name": "Via Expressway", "time": estimated_time, "distance": distance},
            {"name": "Via Sector Roads", "time": estimated_time + random.randint(5, 15), "distance": distance + random.randint(2, 5)},
            {"name": "Via Metro + Walk", "time": random.randint(35, 55), "distance": "N/A"}
        ]
        
        questions = [
            f"How do I go from {source} to {destination}?",
            f"Best route from {source} to {destination}",
            f"Navigate from {source} to {destination}",
            f"What's the fastest way to reach {destination} from {source}?",
            f"Route to {destination} from {source}",
        ]
        
        response = f"""Here are your route options from {source} to {destination}:

**Recommended: {routes[0]['name']}**
- Distance: {routes[0]['distance']} km
- Estimated time: {routes[0]['time']} minutes
- {traffic_note}

**Alternative 1: {routes[1]['name']}**
- Distance: {routes[1]['distance']} km  
- Estimated time: {routes[1]['time']} minutes

**Alternative 2: {routes[2]['name']}**
- Estimated time: {routes[2]['time']} minutes
- Best for avoiding traffic

{'I recommend starting early or using Metro during peak hours.' if time_factor > 1.5 else 'Good time to drive, roads are relatively clear.'}"""
        
        return {
            "instruction": random.choice(questions),
            "response": response,
            "metadata": {
                "type": "route_planning",
                "source": source,
                "destination": destination
            }
        }
    
    def generate_aqi_query_response(self) -> Dict[str, str]:
        """Generate an AQI-related Q&A pair"""
        location = random.choice(self.MAJOR_LOCATIONS[:10])
        
        # Seasonal AQI variation
        month = random.randint(1, 12)
        if month in [11, 12, 1]:  # Winter - worst AQI
            aqi = random.randint(200, 450)
        elif month in [6, 7, 8]:  # Monsoon - best AQI
            aqi = random.randint(50, 120)
        else:
            aqi = random.randint(100, 250)
        
        # Categorize
        if aqi <= 50:
            category, advice = "Good", "Air quality is satisfactory. Perfect for outdoor activities."
        elif aqi <= 100:
            category, advice = "Moderate", "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion."
        elif aqi <= 150:
            category, advice = "Unhealthy for Sensitive Groups", "People with respiratory conditions should reduce outdoor activities."
        elif aqi <= 200:
            category, advice = "Unhealthy", "Everyone may begin to experience health effects. Limit outdoor exposure."
        elif aqi <= 300:
            category, advice = "Very Unhealthy", "Health alert! Everyone should avoid prolonged outdoor exertion."
        else:
            category, advice = "Hazardous", "Emergency conditions. Stay indoors with air purifiers running."
        
        questions = [
            f"What's the AQI at {location}?",
            f"Air quality in {location}",
            f"Is it safe to go outside in {location}?",
            f"Pollution levels at {location}",
            f"How is the air quality near {location}?",
        ]
        
        response = f"""**Air Quality Report for {location}**

Current AQI: {aqi} ({category})

**Health Advisory:**
{advice}

**Traffic's Contribution:** Approximately {random.randint(25, 45)}% of local air pollution comes from vehicular emissions.

**Recommendations:**
- {'Avoid outdoor activities' if aqi > 200 else 'Outdoor activities are fine' if aqi < 100 else 'Limit strenuous outdoor activities'}
- {'Use N95 masks if going out' if aqi > 150 else ''}
- Consider using Metro to reduce emissions
- {'Use air purifiers indoors' if aqi > 200 else ''}"""
        
        return {
            "instruction": random.choice(questions),
            "response": response,
            "metadata": {
                "type": "aqi_query",
                "location": location,
                "aqi": aqi
            }
        }
    
    def generate_scenario_query_response(self) -> Dict[str, str]:
        """Generate a what-if scenario Q&A pair"""
        scenarios = [
            {
                "question": "What if we add a flyover at Sector 62?",
                "impact": "positive",
                "traffic_change": -25,
                "affected_area": "Sector 62 and surrounding sectors"
            },
            {
                "question": "What if DND toll is removed?",
                "impact": "mixed",
                "traffic_change": 40,
                "affected_area": "DND Flyway and Noida-Delhi border"
            },
            {
                "question": "What happens if Aqua Line frequency increases?",
                "impact": "positive",
                "traffic_change": -15,
                "affected_area": "Noida-Greater Noida corridor"
            },
            {
                "question": "Impact of closing NH-24 for repairs?",
                "impact": "negative",
                "traffic_change": 60,
                "affected_area": "All alternative routes to Delhi"
            },
            {
                "question": "What if all autos become electric?",
                "impact": "positive",
                "traffic_change": 0,
                "affected_area": "City-wide AQI improvement"
            }
        ]
        
        scenario = random.choice(scenarios)
        
        response = f"""**Scenario Analysis: {scenario['question']}**

**Impact Assessment:** {scenario['impact'].title()}

**Traffic Impact:**
- Change in congestion: {'+' if scenario['traffic_change'] > 0 else ''}{scenario['traffic_change']}%
- Affected areas: {scenario['affected_area']}

**Analysis:**
{'This would significantly improve traffic flow and reduce commute times.' if scenario['impact'] == 'positive' else 'This would create additional congestion in the short term.' if scenario['impact'] == 'negative' else 'This would have mixed effects - some improvement, some challenges.'}

**Recommendations:**
1. {'Implement in phases to minimize disruption' if scenario['impact'] != 'positive' else 'Fast-track implementation recommended'}
2. Monitor traffic patterns during transition
3. Provide alternative route information to commuters
4. Coordinate with traffic police for management

**Timeline for full effect:** {random.randint(3, 18)} months"""
        
        return {
            "instruction": scenario['question'],
            "response": response,
            "metadata": {
                "type": "scenario_analysis",
                "scenario": scenario['question']
            }
        }
    
    def generate_general_conversation(self) -> Dict[str, str]:
        """Generate general traffic conversation"""
        conversations = [
            {
                "instruction": "Tell me about Noida's traffic patterns",
                "response": """Noida's traffic follows predictable patterns:

**Peak Hours:**
- Morning: 8:00 AM - 11:00 AM (office commute)
- Evening: 5:00 PM - 9:00 PM (return commute)

**Busiest Areas:**
1. Sector 18 (commercial hub)
2. DND Flyway (Delhi connection)
3. NH-24 (major artery)
4. Sector 62 (IT hub)

**Best Times to Travel:**
- Before 7:30 AM
- Between 11:30 AM and 4:30 PM
- After 9:30 PM

**Tips:**
- Use Metro for long distances during peak hours
- Sector roads often less crowded than expressways
- Check traffic before starting (use this assistant!)"""
            },
            {
                "instruction": "How does rain affect traffic in Noida?",
                "response": """Rain significantly impacts Noida's traffic:

**Effects:**
- Speed reduction: 30-50%
- Commute time increase: 40-80%
- Waterlogging in low-lying areas

**Problem Areas During Rain:**
1. Sector 15A underpass
2. Film City road
3. Parts of NH-24
4. Indirapuram Shakti Khand

**Recommendations:**
- Avoid underpasses during heavy rain
- Allow 2x normal travel time
- Check drainage status before travel
- Keep to main roads with better drainage

**Metro Alternative:**
Blue Line and Aqua Line operate normally during rain - highly recommended."""
            },
            {
                "instruction": "What's the best time to travel from Noida to Delhi?",
                "response": """Best times to travel from Noida to Delhi:

**Fastest Times (15-25 min to central Delhi):**
- Before 7:00 AM
- 11:00 AM - 3:00 PM
- After 10:00 PM

**Avoid These Times:**
- 8:00 AM - 11:00 AM (60-90 min journey)
- 5:00 PM - 9:00 PM (45-90 min journey)

**Route Recommendations by Time:**

*Peak Hours:*
- Use Metro (Blue Line) - 30-40 min consistently
- If driving, try Kalindi Kunj instead of DND

*Off-Peak:*
- DND Flyway (fastest, toll ₹40)
- Noida-Delhi Link Road (free, slightly longer)

**Pro Tip:** Friday evenings and Sunday evenings are particularly bad. Plan accordingly!"""
            }
        ]
        
        return random.choice(conversations)
    
    def generate_dataset(self, num_samples: int = 10000) -> List[Dict]:
        """Generate complete training dataset"""
        dataset = []
        
        # Distribution of query types
        distributions = {
            "traffic_query": 0.30,
            "route_planning": 0.25,
            "aqi_query": 0.15,
            "scenario": 0.15,
            "general": 0.15
        }
        
        for _ in range(num_samples):
            rand = random.random()
            cumulative = 0
            
            for query_type, prob in distributions.items():
                cumulative += prob
                if rand <= cumulative:
                    if query_type == "traffic_query":
                        sample = self.generate_traffic_query_response()
                    elif query_type == "route_planning":
                        sample = self.generate_route_query_response()
                    elif query_type == "aqi_query":
                        sample = self.generate_aqi_query_response()
                    elif query_type == "scenario":
                        sample = self.generate_scenario_query_response()
                    else:
                        sample = self.generate_general_conversation()
                    
                    dataset.append(sample)
                    break
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} samples to {output_path}")
    
    def generate_and_save(
        self,
        output_dir: str = "./training_data",
        train_samples: int = 50000,
        val_samples: int = 5000,
        test_samples: int = 5000
    ):
        """Generate and save train/val/test splits"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating training data...")
        train_data = self.generate_dataset(train_samples)
        self.save_dataset(train_data, f"{output_dir}/train.json")
        
        print("Generating validation data...")
        val_data = self.generate_dataset(val_samples)
        self.save_dataset(val_data, f"{output_dir}/val.json")
        
        print("Generating test data...")
        test_data = self.generate_dataset(test_samples)
        self.save_dataset(test_data, f"{output_dir}/test.json")
        
        print(f"\nDataset generation complete!")
        print(f"Train: {len(train_data)} samples")
        print(f"Val: {len(val_data)} samples")
        print(f"Test: {len(test_data)} samples")


# ============================================================================
# ADDITIONAL SPECIALIZED DATA GENERATORS
# ============================================================================

class ConversationGenerator:
    """Generate multi-turn conversations"""
    
    def generate_conversation(self, num_turns: int = 3) -> List[Dict[str, str]]:
        """Generate a multi-turn conversation"""
        conversation = []
        
        # Start with a general query
        user_msg = random.choice([
            "Hi, I need help with traffic in Noida",
            "Can you help me plan my commute?",
            "What's traffic like today?",
        ])
        
        assistant_msg = random.choice([
            "Hello! I'm Traffic God, your Noida traffic assistant. How can I help you today? I can provide traffic updates, route suggestions, and more.",
            "Hi! I'm here to help with traffic in Noida and NCR. What would you like to know?",
        ])
        
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
        
        # Add follow-up turns
        for _ in range(num_turns - 1):
            # User follow-up
            user_msg = random.choice([
                "What about Sector 18?",
                "Is there an alternative route?",
                "What time should I leave?",
                "What about taking the metro?",
            ])
            conversation.append({"role": "user", "content": user_msg})
            
            # Assistant response
            assistant_msg = random.choice([
                f"Sector 18 currently has {random.choice(['moderate', 'heavy'])} traffic. The market area is particularly congested.",
                "Yes! You can take the sector roads instead - it adds 2 km but saves about 10 minutes during peak hours.",
                "I recommend leaving before 8 AM or after 11 AM to avoid the worst traffic.",
                "Metro is a great option! The Blue Line runs every 4-6 minutes during peak hours.",
            ])
            conversation.append({"role": "assistant", "content": assistant_msg})
        
        return conversation


if __name__ == "__main__":
    # Generate training data
    generator = TrafficDataGenerator(seed=42)
    generator.generate_and_save(
        output_dir="./new_traffic_god/training_data",
        train_samples=50000,
        val_samples=5000,
        test_samples=5000
    )
