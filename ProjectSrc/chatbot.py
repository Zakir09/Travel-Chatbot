from dotenv import load_dotenv
import os
import spacy
from spacy.matcher import Matcher
from nltk.corpus import wordnet as wn
import requests
import opencage.geocoder
from openai import OpenAI
import random

# Load environment variables from .env file
load_dotenv()

# Use environment variables
open_ai_key = os.getenv('OPEN_AI_KEY')
yelp_api_key = os.getenv('YELP_API_KEY')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
opencage_api_key = os.getenv('OPENCAGE_API_KEY')

client = OpenAI(
    api_key=open_ai_key,
)

nlp = spacy.load("en_core_web_lg")


class TravelPlanState:
    def __init__(self):
        self.location = None
        self.food_preferences = []
        self.activity_preferences = []
        self.hobby_preferences = []
        self.num_days = 1
        self.meal_options = {}
        self.activity_options = []
        self.hotel_options = []

    def update_preferences(self, location=None, food_preferences=None, activity_preferences=None, hobby_preferences=None, num_days=None):
        if location:
            self.location = location
        if activity_preferences:
            self.activity_preferences.extend(activity_preferences)  # Use extend instead of append
        if food_preferences:
            self.food_preferences.extend(food_preferences)  # Use extend instead of append
        if hobby_preferences:
            self.hobby_preferences.extend(hobby_preferences)
        if num_days:
            self.num_days = num_days

    def update_plan(self, meal_options=None, activity_options=None, hotel_options=None):
        if meal_options:
            self.meal_options = meal_options
        if activity_options:
            self.activity_options = activity_options
        if hotel_options:
            self.hotel_options = hotel_options

    def clear_preferences(self):
        self.food_preferences = []
        self.activity_preferences = []
        self.hobby_preferences = []

    def reset(self):
        self.__init__()

travel_plan_state = TravelPlanState()


def get_lemmas(word):
    lemmas = set()
    for syn in wn.synsets(word, pos=wn.NOUN):  # Consider using other POS tags if relevant
        for lemma in syn.lemmas():
            lemmas.add(lemma.name().replace('_', ' ').lower())
    return lemmas

def get_extended_synonyms(word):
    manual_synonyms = {
        "outdoor": {"outdoors", "exploring", "nature", "hiking"},
        "watersport": {"watersports", "water sport", "water sports"},
        "winter": {"cold", "snow", "ice"},
        "cycling": {"biking", "bike", "bicycle"},
        "fitness": {"exercise", "workout", "health", "gym"},
        "photo": {"photography", "photographing", "photos"},
        "nature": {"natural", "outdoors", "environment", "wildlife"},
        "culture": {"cultural", "historical", "heritage", "arts"},
        "music": {"musical", "concerts", "singing", "live music"},
        "game": {"gaming", "games", "video games", "arcades"},
        "excite": {"adventurous", "explore", "exploration", "thrill"},
        "leisure": {"relaxation", "relaxing", "chill", "leisurely", "relax"},
        "creative": {"creativity", "create", "craft", "arts and crafts", "creating"},
        "culinary": {"cooking", "cook", "foodie", "gourmet"},
        "entertainment": {"amusement", "fun", "entertaining"},
        "exploration": {"explore", "discovery", "adventure", "travel"},
        "relaxation": {"relax", "peaceful", "rest", "chill"},
        "party": {"nightlife", "clubbing", "partying"},
        "water": {"aquatic", "sea", "ocean", "marine"},
        # Add more as needed for each hobby key
    }

    synonyms = get_lemmas(word)  # Your existing lemma generation
    manual_related = manual_synonyms.get(word, set())
    return synonyms.union(manual_related)

def extract_information(user_input):
    doc = nlp(user_input)
    matcher = Matcher(nlp.vocab)

    # Extract location
    location = None
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE stands for geopolitical entity
            location = ent.text
            if location == "uk":
                location = None
            break

     # Combine token texts to catch multi-word preferences
    text = [token.text.lower() for token in doc]
    combined_text = ' '.join(text)

    food_preferences = set()
    activity_preferences = set()
    hobby_preferences = set()

    # Define multi-word and single-word preferences
    food_keywords = [
        "italian", "chinese", "mexican", "indian", "japanese", "thai", "french", "mediterranean",
        "american", "bbq", "vegan", "vegetarian", "seafood", "steakhouse", "cafe", "buffet",
        "pub", "fast food", "food truck", "fusion", "gourmet", "tapas", "diner", "sushi",
        "halal", "korean", "vietnamese", "spanish", "greek", "lebanese", "turkish",
        "ethiopian", "caribbean", "peruvian", "brazilian", "african", "european",
        "gluten free", "bakery", "patisserie", "gelato", "ice cream", "dessert", "coffee",
        "tea", "bistro", "pizzeria", "burger", "pasta", "dim sum", "noodle", "ramen",
        "taco", "burrito", "pizza", "kebab", "falafel", "soul food", "farm to table",
        "polish", "russian", "belgian", "argentinian", "chilean", "salvadoran", "cuban",
        "filipino", "indonesian", "malaysian", "mongolian", "moroccan", "persian",
        "scandinavian", "swiss", "ukrainian", "venezuelan", "gluten-free", "lactose-free",
        "paleo", "raw", "smoothies", "juice bar", "organic", "locally sourced",
        "health food", "ayurvedic", "kosher", "middle eastern", "nordic", "portuguese",
        "sri lankan", "tibetan", "welsh", "scottish", "irish", "hawaiian", "jamaican",
        "raw vegan", "whole food", "low carb", "pescatarian", "flexitarian"
    ]

    hobby_to_activity_map = {
        "outdoor": ["hike", "camping", "glamping", "kayaking", "sailing", "fishing", "rock climbing", "mountain biking", "bird watching", "nature walk", "paddleboarding", "wild swimming", "treasure hunt"],
        "watersport": ["kayaking", "canoeing", "sailing", "fishing", "paddleboarding", "wild swimming"],
        "winter": ["skiing", "snowboarding", "ice skating"],
        "cycling": ["cycling", "mountain biking", "biking"],
        "fitness": ["yoga", "pilates", "fitness class", "running", "sports event"],
        "photo": ["photography", "scenic view", "monument", "landmark", "architecture tour", "photography class"],
        "nature": ["nature walk", "bird watching", "botanical garden", "reservoir", "garden visit"],
        "culture": ["museum", "art gallery", "cultural center", "street art", "mural", "cultural workshop", "castle tour", "literary tour", "film tour"],
        "music": ["concert", "dance class", "karaoke", "jazz club", "live music venue", "open mic night"],
        "game": ["cinema", "arcade", "virtual reality", "trampoline park", "skate park", "escape room", "laser tag", "bowling", "mini golf", "go-karting", "board game café", "virtual escape room", "immersive experience", "VR arcade", "video game lounge"],
        "excite": ["rock climbing", "aerial adventure", "zip lining", "paintball", "bumper cars", "escape game", "airsoft", "hovercraft experience", "zorbing", "adventure"],
        "leisure": ["picnic spot", "sunset watch", "dawn patrol", "roller skating", "tea tasting", "coffee tasting", "beach", "park", "spa"],
        "creative": ["pottery class", "painting class", "drawing class", "sculpture class", "photography class", "animation workshop", "craft workshop", "DIY pottery studio", "soap making workshop", "perfume making workshop"],
        "culinary": ["cooking class", "winery", "brewery", "tea tasting", "coffee tasting", "cheese tasting", "chocolate workshop", "wine tasting", "beer tasting", "distillery tour", "cider tasting", "sushi making class", "chocolate making class", "bread baking workshop", "cocktail making class"],
        "entertainment": ["theme park", "amusement park", "casino", "dinner theater", "comedy club", "magic show", "puppet show", "murder mystery dinner", "planetarium"],
        "exploration": ["historical site", "zoo", "aquarium", "castle tour", "ghost tour", "literary tour", "film tour", "farm visit", "market", "antique hunting", "treasure hunt", "sailing experience"],
        "relaxation": ["spa", "beach", "park", "ice bar", "rooftop cinema", "drive-in movie", "silent disco"],
        "party": ["bar", "nightclub", "karaoke", "jazz club", "comedy club", "live music venue", "open mic night", "silent disco"],
        "water": ["aquarium", "beach", "lake", "river", "waterfall", "paddleboarding", "wild swimming"],
    }


    activity_keywords = [
        "hike", "shopping", "museum", "theme park", "beach", "concert", "cinema", 
        "historical site", "zoo", "art gallery", "bar", "nightclub", "spa", "golf", 
        "amusement park", "aquarium", "botanical garden", "casino", "cultural center", 
        "observatory", "winery", "brewery", "cruise", "festival", "lake", "river", 
        "waterfall", "rock climbing", "kayaking", "canoeing", "sailing", "fishing", 
        "skiing", "snowboarding", "ice skating", "horse riding", "camping", "glamping", 
        "nature walk", "bird watching", "cycling", "mountain biking", "yoga", "pilates", 
        "fitness class", "sports event", "football", "basketball", "baseball", "soccer", 
        "tennis", "running", "marina", "sunset watch", "photography", "picnic", "scenic view", 
        "monument", "landmark", "street art", "mural", "workshop", 
        "cooking class", "dance class", "language class", "escape room", "virtual reality", 
        "arcade", "trampoline", "skateboarding", "roller skating", "adventure", "zip lining",
        "paddleboarding", "wild swimming", "garden visit", "castle tour", "pottery class",
        "tea tasting", "coffee tasting", "cheese tasting", "chocolate workshop", "ghost tour",
        "literary tour", "film tour", "farm visit", "market", "antique hunting", "craft fair",
        "biking", "painting class", "drawing class", "sculpture class", "photography class",
        "distillery tour", "cider tasting", "laser tag", "bowling", "mini golf", "go-karting", "bumper cars", "board game café",
        "paintball", "aquapark", "roller disco", "karaoke", "bouldering", "indoor skydiving",
        "theme restaurant", "magic show", "puppet show", "ice bar", "dinner theater",
        "comedy club", "jazz club", "planetarium", "murder mystery dinner", "virtual reality",
        "axe throwing", "rooftop cinema", "drive-in movie", "immersive experience", "VR",
        "bubble soccer", "indoor surfing", "hovercraft experience", "zorbing", "treasure hunt",
        "escape game", "airsoft", "live music venue", "open mic night", "silent disco",
        "nerf battle", "drone racing", "video game lounge", "animation workshop", "sailing experience",
        "craft workshop", "DIY pottery studio", "soap making workshop", "perfume making workshop",
        "cocktail making class", "sushi making class", "chocolate making class", "bread baking workshop"
    ]


    # Lemmatize user input
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    for hobby, activities in hobby_to_activity_map.items():
        hobby_synonyms = get_extended_synonyms(hobby)
        if any(hobby_syn in lemmatized_text for hobby_syn in hobby_synonyms):
            selected_activity = random.choice(activities)
            activity_preferences.add(selected_activity)
            selected_hobby = hobby
            hobby_preferences.add(selected_hobby)

    # Check for multi-word and single-word preferences in combined_text
    for preference in food_keywords + activity_keywords:
        if preference in combined_text:
            if preference in food_keywords:
                food_preferences.add(preference)
            else:
                activity_preferences.add(preference)

    food_preferences = list(food_preferences)
    activity_preferences = list(activity_preferences)
    hobby_preferences = list(hobby_preferences)
    if food_preferences:
        travel_plan_state.update_preferences(food_preferences=food_preferences)
    if activity_preferences:
        travel_plan_state.update_preferences(activity_preferences=activity_preferences)
    if hobby_preferences:
        travel_plan_state.update_preferences(hobby_preferences=hobby_preferences)

    # Default random selection if no preferences are found
    if not food_preferences and not travel_plan_state.food_preferences and location:
        food_preferences.append(random.choice(food_keywords))
        food_preferences.append(random.choice(food_keywords))
        print (food_preferences)
    if not activity_preferences and not travel_plan_state.activity_preferences and location:
        activity_preferences.append(random.choice(activity_keywords))
        print (activity_preferences)

   # Patterns to match "[number] day(s)" or "[number in text] day(s)"
    patterns = [
        [{"SHAPE": "d"}, {"LOWER": "day"}, {"IS_PUNCT": True, "OP": "?"}],
        [{"SHAPE": "d"}, {"LOWER": "days"}, {"IS_PUNCT": True, "OP": "?"}],
        [{"LOWER": {"IN": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]}}, {"LOWER": "day"}, {"IS_PUNCT": True, "OP": "?"}],
        [{"LOWER": {"IN": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]}}, {"LOWER": "days"}, {"IS_PUNCT": True, "OP": "?"}],
    ]
    matcher.add("NumDays", patterns)

    # Mapping words to numeric values
    word_to_number = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    matches = matcher(doc)
    num_days = travel_plan_state.num_days

    for match_id, start, end in matches:
        span = doc[start:end-1]  # Exclude the optional punctuation
        text = span.text.lower().split()[0]  # Get the number part of the match
        
        if text.isdigit():
            num_days = int(text)
        elif text in word_to_number:
            num_days = word_to_number[text]
        break  # Assuming the first match is the desired one; remove or modify if multiple day specifications can occur.
    travel_plan_state.update_preferences(num_days=num_days)

    return location, num_days, food_preferences, activity_preferences, hobby_preferences



def get_yelp_data(query, location, limit=40, radius=10000):
    # Make a request to the Yelp API
    headers = {
        "Authorization": f"Bearer {yelp_api_key}"
    }

    # Update the parameters to include the specific location and the increased limit
    params = {
        "term": query,
        "location": location,
        "limit": limit,
        "radius": radius
    }

    endpoint = "https://api.yelp.com/v3/businesses/search"
    response = requests.get(endpoint, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        businesses = data.get("businesses", [])
        return businesses
    else:
        return None
    

def get_activity_options(activity_preferences, location, num_days):
    # Function to remove duplicates based on Yelp business ID
    def remove_duplicates(places):
        unique_places = {}
        for place in places:
            if place['id'] not in unique_places:
                unique_places[place['id']] = place
        return list(unique_places.values())
    
    def ensure_minimum_activities(activity_preferences, min_count=4):
        general_activities = [
            "hiking", "shopping", "theme park", "concert", "cinema", 
            "historical site", "art gallery", "spa", "golf", 
            "amusement park", "aquarium", "botanical garden", 
            "cruise", "festival", "rock climbing", "kayaking", "canoeing", "sailing", 
            "ice skating", "horse riding", "camping", 
            "cycling", "mountain biking", "photography", 
            "scenic view", "landmark", "street art", "mural", "workshop", 
            "escape room", "virtual reality", "arcade", "trampoline", "skateboarding", 
            "roller skating", "paddleboarding", "wild swimming", "garden visit", 
            "castle tour", "ghost tour", "market",
            "laser tag", "bowling", "mini golf", "go-karting", "bumper cars", "board game café",
            "paintball", "aquapark", "roller disco", "karaoke", "bouldering", "indoor skydiving",
            "theme restaurant", "magic show", "puppet show", "ice bar", "dinner theater",
            "comedy club", "jazz club", "planetarium", "murder mystery",
            "axe throwing", "rooftop cinema", "drive-in movie", "immersive experience", "vr",
            "bubble soccer", "indoor surfing", "hovercraft experience", "zorbing", "treasure hunt",
            "escape game", "airsoft", "live music venue", "open mic night", "silent disco",
            "nerf battle", "drone racing", "video game lounge"
        ]


        while len(activity_preferences) < min_count * (num_days / 2):
            # Add a random activity that is not already in the preferences
            additional_activity = random.choice([activity for activity in general_activities if activity not in activity_preferences])
            activity_preferences.append(additional_activity)
        return activity_preferences


    def spread_activities(activities, num_days):
        activity_types = list(activities.keys())  # List of different activity types
        current_activity_index = 0  # Track the current index of activity types
        activities_per_day = 3  # Desired number of activities per day
        activity_selection.clear()  # Clearing existing selections if you're reusing this list

        total_activities_needed = num_days * activities_per_day
        selected_activities_count = 0

        while selected_activities_count < total_activities_needed:
            if current_activity_index >= len(activity_types):  # If we've gone through all types, start over
                current_activity_index = 0

            current_type = activity_types[current_activity_index]
            if activities[current_type]:  # If there are places left in this activity type
                selected_place = random.choice(activities[current_type])  # Randomly select a place
                print(f"Selected activity place: {selected_place['name']} ({current_type})")
                
                activity_selection.append(selected_place)  # Add to the global activity_selection list
                selected_activities_count += 1

            current_activity_index += 1

    
    # Ensure a minimum number of activities
    activity_preferences_temp = list(activity_preferences)
    activity_preferences_temp = ensure_minimum_activities(activity_preferences_temp)
    print(activity_preferences_temp)

    # Group activities by type
    activity_groups = {preference: [] for preference in activity_preferences_temp}

    for preference in activity_preferences_temp:
        query = f"{preference}"
        print(f"Fetching data for: {query}, Location: {location}")
        places = get_yelp_data(query, location)  # Adjust limit as needed
        if places:
            print(f"Found {len(places)} places for query: {query}")
            activity_groups[preference] = remove_duplicates(places)
        else:
            print(f"No places found for query: {query}")


    activity_selection = []
    spread_activities(activity_groups, num_days)

    return activity_selection


def get_meal_options(food_preferences, location, num_days):
    meal_options = {"breakfast": [], "lunch": [], "dinner": []}
    
    # Function to remove duplicates based on Yelp business ID
    def remove_duplicates(places):
        unique_places = {}
        for place in places:
            if place['id'] not in unique_places:
                unique_places[place['id']] = place
        return list(unique_places.values())

    for meal_type in ["breakfast", "lunch", "dinner"]:
        all_places = []  # Collect all places for all preferences for each meal

        # Loop through each food preference and make separate requests,
        # explicitly appending "restaurant" to each preference to form the query
        for preference in food_preferences:
            # Append "restaurant" to make the preference explicit for Yelp API
            query = f"{preference} {meal_type}"
            print(f"{preference} restaurant {meal_type} day {num_days}")
            print(f"Fetching data for: {query}, Location: {location}")
            places = get_yelp_data(query, location)  # Adjust limit as needed
            if places:
                print(f"Found {len(places)} places for query: {query}")
                all_places.extend(places)
            else:
                print(f"No places found for query: {query}")
            
            # Remove duplicates and randomly select one unique place for the meal
            unique_places = remove_duplicates(all_places)
        for day in range(1, num_days + 1):
            if unique_places:
                selected_place = random.choice(unique_places)  # Randomly select one unique place
                print(f"Selected place for {meal_type} on day {day}: {selected_place['name']}")
                meal_options[meal_type].append(selected_place)
    
    return meal_options


def get_coordinates(location):
    # Use OpenCage Geocoding API to get coordinates for the location
    geocoding_endpoint = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        "q": location,
        "key": opencage_api_key
    }

    response = requests.get(geocoding_endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            # Extract latitude and longitude
            lat = data["results"][0]["geometry"]["lat"]
            lon = data["results"][0]["geometry"]["lng"]
            return f"{lat},{lon}"
        else:
            return None
    else:
        return None

def get_hotel_data(location, activities, food_places):
    # Get coordinates for the location
    coordinates = get_coordinates(location)

    if coordinates:
        # Make a request to the Google Maps Places API for hotels
        endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": coordinates,
            "radius": 4000,  # Adjust the radius based on your preference
            "type": "lodging",
            "key": google_maps_api_key
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            hotels = data.get("results", [])

            # Score hotels based on ratings, affordability, and proximity
            scored_hotels = score_hotels(hotels, activities, food_places)

            # Sort hotels by score (higher score is better)
            sorted_hotels = sorted(scored_hotels, key=lambda x: x["score"], reverse=True)

            return sorted_hotels
        else:
            return None
    else:
        print("Error getting coordinates for the location.")
        return None

def score_hotels(hotels, activities, food_places):
    scored_hotels = []

    for hotel in hotels:
        # Assign scores based on ratings, affordability, and proximity
        rating_score = hotel.get("rating", 0)
        affordability_score = 1  # Placeholder, you can adjust this based on your criteria
        proximity_score = calculate_proximity_score(hotel["geometry"]["location"], activities, food_places)

        # Total score (you can adjust the weights based on your preference)
        total_score = 0.6 * rating_score + 0.2 * affordability_score + 0.2 * proximity_score

        scored_hotels.append({"hotel": hotel, "score": total_score})

    return scored_hotels

def calculate_proximity_score(hotel_location, activities, food_places):
    # Placeholder function, you can implement logic to calculate proximity score based on distances
    return 1.0  # Adjust based on your criteria

def is_location_in_uk(location):
    geocoder = opencage.geocoder.OpenCageGeocode(opencage_api_key)
    result = geocoder.geocode(location)
    for res in result:
        if res['components']['country'] == 'United Kingdom':
            return True
    return False



def chat_gpt(messages):
    user_input = messages.lower()
    location, num_days, food_preferences, activity_preferences, hobby_preferences = extract_information(user_input)

    # Check if the input is a general conversation
    if (not location or location == travel_plan_state.location) and (not food_preferences or not activity_preferences or not hobby_preferences):
        # Handle general conversation using GPT-4
        system_message = {"role": "system", "content": "You are a versatile travel chatbot capable of engaging in general conversations and providing helpful responses to users about questions on cities/places in the UK."}
        user_message = {"role": "user", "content": user_input}
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[system_message, user_message],
                temperature=0.7  # Adjust for creativity as needed
            )
            chat_response = response.choices[0].message.content
            return chat_response
        except Exception as e:
            return f"Sorry, I encountered an issue: {str(e)}"
    if location:
        travel_plan_state.update_preferences(location=location)
    if not location and travel_plan_state.location:
        location = travel_plan_state.location
    num_days = travel_plan_state.num_days or 1
    if food_preferences and travel_plan_state.food_preferences:
        food_preferences = travel_plan_state.food_preferences
    if activity_preferences and travel_plan_state.activity_preferences:
        activity_preferences = travel_plan_state.activity_preferences
    if hobby_preferences and travel_plan_state.hobby_preferences:
        hobby_preferences = travel_plan_state.hobby_preferences
    
    # Proceed with location-specific logic for travel-related inquiries
    if location:
        if not is_location_in_uk(location):
            system_message = {"role": "system", "content": "You are a versatile travel chatbot capable of engaging in general conversations and providing helpful responses to users about questions on places."}
            user_message = {"role": "user", "content": "Tell the user that they need to include a destination within the UK and provide popular places."}
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[system_message, user_message],
                    temperature=0.7  # Adjust for creativity as needed
                )
                chat_response = response.choices[0].message.content
                return chat_response
            except Exception as e:
                return f"Sorry, I encountered an issue: {str(e)}"
        else:
            location += ", UK"
    else:
        system_message = {"role": "system", "content": "You are a versatile travel chatbot capable of engaging in general conversations and providing helpful responses to users about questions on places."}
        user_message = {"role": "user", "content": "Tell the user that they need to include a destination within the UK and provide popular places"}
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[system_message, user_message],
                temperature=0.7  # Adjust for creativity as needed
            )
            chat_response = response.choices[0].message.content
            return chat_response
        except Exception as e:
            return f"Sorry, I encountered an issue: {str(e)}"

    print(f"{location}\n{food_preferences}\n{activity_preferences}\n{hobby_preferences}\n{num_days}")

    meal_options = get_meal_options(food_preferences, location, num_days)
    all_activities = get_activity_options(activity_preferences, location, num_days)

    if not all_activities:
        return "I'm sorry, but I couldn't find any activity options for your trip based on the given preferences."

    # Use the length of all_activities list to check if there are enough activities
    if len(all_activities) < 2 * num_days:
        return "I'm sorry, but I couldn't find enough diverse activity options for your trip."
    # The rest of your function...

    hotels = get_hotel_data(location, all_activities, meal_options["breakfast"] + meal_options["lunch"] + meal_options["dinner"])
    travel_plan_state.update_plan(activity_options=all_activities, meal_options=meal_options, hotel_options=hotels)
    

    # Construct a detailed prompt for GPT
    prompt_summary = f"Create a detailed travel itinerary for a {num_days}-day trip to {location}. Include dining at cuisines and restaurants, activities the user could do, and recommended hotels. Consider the following details:\n"
    prompt_summary = f"\nHobbies/Interests:\n"
    prompt_summary = f"\n{hobby_preferences}\n"
    prompt_summary += "\nMeal options:\n"
    for meal_type, meals in meal_options.items():
        prompt_summary += f"{meal_type.capitalize()}: " + ", ".join([f"{meal['name']}" for meal in meals]) + "\n"

    prompt_summary += "\nActivities:\n"
    activities_per_day = len(all_activities) // num_days
    for day in range(num_days):
        activities_for_day = all_activities[day * activities_per_day: (day + 1) * activities_per_day]
        prompt_summary += f"Day {day + 1}: " + ", ".join([activity['name'] for activity in activities_for_day]) + "\n"

    prompt_summary += "\nHotel recommendations:\n" + ", ".join([hotel['hotel']['name'] for hotel in hotels[:5]]) + "\n"
    prompt_summary += "\nHere is also the original message(s) from the user to take into account:\n"
    prompt_summary += f"\n{user_input}"
    prompt_summary += "\nPlease organize this information into a coherent, long, descriptive, and detailed travel plan.\n"
    prompt_summary += "\nMake sure it is formatted as if you were consulting someone."

    # Call to GPT-4
    system_message = {"role": "system", "content": "You are a helpful travel agent designed to provide travel advice and recommendations in a destination in the UK."}
    user_message = {"role": "user", "content": prompt_summary}

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[system_message, user_message],
            temperature=0.7  # Adjust for creativity as needed
        )
        chat_response = response.choices[0].message.content
    except Exception as e:
        chat_response = f"Sorry, I encountered an issue: {str(e)}"
    
    return chat_response


