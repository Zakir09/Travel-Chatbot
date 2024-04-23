from dotenv import load_dotenv
import os
import spacy
from spacy.matcher import Matcher
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import requests
import opencage.geocoder
from openai import OpenAI
import random
# import gc


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

nlp = spacy.load("updated_ner_model")
geocoder = opencage.geocoder.OpenCageGeocode(opencage_api_key)


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
        "hiking": {"hike", "trails", "trekking", "walking", "exploring", "explore"},
        "art": {"art", "painting", "sculpture", "galleries", "creativity"},
        "literature": {"books", "reading", "literature", "novels", "writing"},
        "technology": {"tech", "gadgets", "innovation", "science", "electronics"},
        "gardening": {"plants", "gardening", "horticulture", "flowers", "landscaping"},
        "astronomy": {"stars", "planets", "astronomy", "space", "universe"},
        "animal": {"animal", "pet", "wildlife", "zoo", "creatures"},
        "water sports": {"surfing", "diving", "swimming", "watersports", "aquatic"},
        "extreme sports": {"extreme", "adrenaline", "adventure sports", "thrill seeking", "action sports"},
        "martial arts": {"martial arts", "fighting", "self-defense", "combat", "dojo"},
        "dance": {"dance", "dancing", "ballet", "salsa", "rhythm"},
        "history": {"history", "past", "historical", "ancient", "heritage"},
        "architecture": {"buildings", "design", "architecture", "construction", "structures"},
        "fashion": {"fashion", "style", "clothing", "apparel", "trends"},
        "wellness": {"wellness", "health", "spa", "relaxation", "self-care"},
        "adventure": {"adventure", "explore", "discovery", "excursion", "journey"},
        "socializing": {"social", "community", "friends", "networking", "gatherings"},
        "mindfulness": {"mindfulness", "meditation", "peace", "calm", "serenity"},
        "eco friendly": {"eco", "sustainability", "environment", "green", "conservation"},
        "music": {"music", "concerts", "instruments", "bands", "melody"},
        "photography": {"photography", "photo", "cameras", "picture", "imagery"},
        "traveling": {"travel", "exploration", "adventures", "journeys", "voyages"},
        "fitness": {"fitness", "exercise", "workout", "gym", "physical"},
        "cycling": {"cycling", "biking", "bicycling", "riding", "pedaling"},
        "board games": {"board games", "tabletop games", "strategy games", "card games", "puzzles"},
        "game": {"gaming", "games", "video games", "arcades"},
        "cooking": {"cooking", "culinary", "baking", "gastronomy", "chef"},
        "film": {"films", "movies", "cinema", "screenings", "blockbusters"},
        "theater": {"theater", "drama", "plays", "performances", "acting"},
        "nature": {"nature", "outdoors", "wilderness", "scenery", "natural", "environment"}
    }

    synonyms = get_lemmas(word)  # Your existing lemma generation
    manual_related = manual_synonyms.get(word, set())
    return synonyms.union(manual_related)

def determine_intent(lemmatized_text):
    intent_keywords = {
        'advice': ['advice', 'info', 'information', 'details', 'tell me about', 'what to do in', 'recommend', 'suggestions'],
        'plan_trip': ['plan', 'trip plan', 'trip', 'schedule', 'organize', 'arrange', 'itinerary']
    }
    for intent, keywords in intent_keywords.items():
        if any(keyword in lemmatized_text for keyword in keywords):
            return intent
    return None  # Default to None if no specific intent is detected


def extract_information(user_input):
    print("Processing input:", user_input)  # To see exactly what is being processed
    doc = nlp(user_input)
    
    matcher = Matcher(nlp.vocab)

    print("Recognized entities and their labels:")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")
    # Extract location
    location = travel_plan_state.location
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE stands for geopolitical entity
            location = ent.text
            if location == "uk" or not is_location_in_uk(location):
                location = travel_plan_state.location 
            if is_location_in_uk(location):
                travel_plan_state.update_preferences(location=location)
                break
    
    if location:
        print(f"Extracted location: {location}")
    else:
        print("No suitable location found.")

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
        "bistro", "pizzeria", "burger", "pasta", "dim sum", "noodle", "ramen",
        "taco", "burrito", "pizza", "kebab", "falafel", "soul food", "farm to table",
        "polish", "russian", "belgian", "argentinian", "chilean", "salvadoran", "cuban",
        "filipino", "indonesian", "malaysian", "mongolian", "moroccan", "persian",
        "scandinavian", "swiss", "ukrainian", "venezuelan", "lactose free",
        "paleo", "raw", "smoothies", "juice bar", "organic", "locally sourced",
        "health food", "ayurvedic", "kosher", "middle eastern", "nordic", "portuguese",
        "sri lankan", "tibetan", "welsh", "scottish", "irish", "hawaiian", "jamaican",
        "raw vegan", "whole food", "low carb", "pescatarian", "flexitarian"
    ]

    hobby_to_activity_map = {
        "hiking": ["trail walking", "mountain trekking", "nature trails", "forest hiking"],
        "art": ["gallery visit", "art exhibition", "museum tour", "street art tour"],
        "literature": ["library", "author meetup", "bookstore", "poetry cafe", "historical library", "literary museum"],
        "technology": ["electronics store", "tech workshops", "tech exhibit", "science museums", "virtual reality", "tech exhibits"],
        "gardening": ["botanical gardens", "community gardening", "floral workshops", "garden center", "flower shop"],
        "astronomy": ["planetarium", "observatory", "astronomy museum", "telescope shop"],
        "animal": ["zoo", "aquarium", "pet cafe", "wildlife rescue center", "bird sanctuary"],
        "water sports": ["surf shop", "dive center", "kayaking location", "water park"],
        "extreme sports": ["indoor skydiving center", "bungee jump site", "paragliding", "mountain bike trail", "rock climbing gym"],
        "martial arts": ["martial arts gym", "karate classes", "judo workshops", "taekwondo", "Brazilian jiu-jitsu", "boxing gym", "aikido dojo"],
        "dance": ["dance studio", "nightclub", "tango club", "salsa night", "ballroom dancing", "street dance workshops"],
        "history": ["history museum", "historical landmark", "cultural heritage center", "archaeological sites", "heritage tours"],
        "architecture": ["architectural tour", "historical building visits", "modern architecture tours"],
        "fashion": ["fashion shows", "fashion boutique", "vintage clothing store", "designer store", "style workshops"],
        "wellness": ["day spa", "yoga studio", "meditation center", "health retreat", "hot spring", "wellness center"],
        "adventure": ["adventure parks", "escape rooms", "theme park", "outdoor adventure course", "expedition tours"],
        "socializing": ["food fairs", "outdoor concerts", "live music venue", "bars", "nightclubs", "conventions"],
        "mindfulness": ["meditation sessions", "mindfulness retreats", "yoga classes", "well-being workshops"],
        "eco friendly": ["nature reserve", "eco park", "organic farm", "sustainability center", "wildlife refuge", "conservation area"],
        "music": ["concerts", "open mic night", "vinyl collecting", "music venue", "jazz club", "record store", "karaoke bar", "music equipment store"],
        "photography": ["photography studio", "gallery", "photo tour", "camera store", "nature photography", "urban photography spot", "photography exhibitions"],
        "traveling": ["road trips", "cultural exchanges", "cultural landmark", "national park", "historical tour", "cruise line", "city tour"],
        "fitness": ["gym", "crossfit box", "pilates studio", "running track", "sports complex", "obstacle course"],
        "cycling": ["bike shop", "cycling trail", "mountain biking park", "bicycle rental service", "BMX track"],
        "game": ["cinema", "arcade", "trampoline park", "skate park", "escape room", "laser tag", "bowling", "mini golf", "go-karting", "board game cafe", "immersive experience", "virtual reality", "video game lounge"],
        "board games": ["board game shop", "game cafe", "gaming lounge", "tabletop gaming club", "puzzle room", "strategy tournament", "RPG session", "card game competition"],
        "cooking": ["cooking class", "food festival", "gourmet shop", "kitchen supply store", "culinary event space", "farmer's market", "food tasting event"],
        "film": ["cinema", "outdoor cinema", "live theater", "musical performance", "comedy show", "opera house", "movie theater", "drive-in cinema", "film studio tour"],
        "theater": ["theater venue", "performing arts center", "opera house", "improvisational theater", "puppet theater", "amphitheater"],
        "nature":["lakefront", "river path", "waterfall trail", "nature preserve", "bird sanctuary", "botanical garden", "reservoir area"]
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
        "distillery tour", "cider tasting", "laser tag", "bowling", "mini golf", "go karting", "bumper car", "board game cafe",
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

    user_intent = determine_intent(lemmatized_text)

    for hobby, activities in hobby_to_activity_map.items():
        hobby_synonyms = get_extended_synonyms(hobby)
        if any(hobby_syn in lemmatized_text for hobby_syn in hobby_synonyms):
            selected_activity = random.choice(activities)
            activity_preferences.add(selected_activity)
            selected_hobby = hobby
            hobby_preferences.add(selected_hobby)

    for preference in food_keywords + activity_keywords:
        if preference in combined_text:
            if preference in food_keywords and preference not in travel_plan_state.food_preferences:
                food_preferences.add(preference)
            elif preference in activity_keywords and preference not in travel_plan_state.activity_preferences:
                activity_preferences.add(preference)

    # Convert from set to list before updating to maintain consistency with your TravelPlanState structure
    food_preferences = list(food_preferences)
    activity_preferences = list(activity_preferences)
    hobby_preferences = list(hobby_preferences)

    # Update travel plan state only with new preferences
    if food_preferences:
        travel_plan_state.update_preferences(food_preferences=food_preferences)
    if activity_preferences:
        travel_plan_state.update_preferences(activity_preferences=activity_preferences)
    if hobby_preferences:
        travel_plan_state.update_preferences(hobby_preferences=hobby_preferences)

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

    # del nlp
    # gc.collect()

    return location, user_intent


def get_yelp_data(query, location, min_rating = 4.2, limit=40, radius=10000):
    headers = {
        "Authorization": f"Bearer {yelp_api_key}"
    }
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
        
        # Filter businesses by minimum rating
        filtered_businesses = [business for business in businesses if business.get("rating", 0) >= min_rating]
        
        return filtered_businesses
    else:
        return None

def remove_duplicates(places):
        unique_places = {}
        for place in places:
            if place['id'] not in unique_places:
                unique_places[place['id']] = place
        print(f"Removed duplicates: {len(unique_places)} unique places found after processing.")
        return list(unique_places.values())

def get_activity_options(activity_preferences, location, num_days):
    general_search_term = "activities"
    print("Starting to fetch general activities...")  # Initial debug statement
    general_activities = get_yelp_data(general_search_term, location, limit=40)
    
    if general_activities:
        print(f"Found {len(general_activities)} general activities.")
    else:
        print("No general activities found.")
    
    specific_activities = []
    general_activities = remove_duplicates(general_activities)  # Apply removal of duplicates and get the result

    for preference in activity_preferences:
        print(f"\nProcessing activity preference: {preference}")
        if preference.lower() != general_search_term:
            specific_results = get_yelp_data(preference, location, limit=40)
            if specific_results:
                print(f"Found {len(specific_results)} specific activities for {preference}.")
                specific_activities.extend(specific_results)
            else:
                print(f"No specific activities found for {preference}.")

    specific_activities = remove_duplicates(specific_activities)  # Remove duplicates from specific results

    # Additional debug statements for summary
    print(f"\nTotal general activities retained: {len(general_activities)}")
    print(f"Total specific activities retained: {len(specific_activities)}")

    # Shuffle for diversity
    random.shuffle(general_activities)
    random.shuffle(specific_activities)

    activity_selection = []
    activities_per_day = 3  # Set the desired number of activities per day

    for day in range(num_days):
        today_activities = []
        print(f"\nScheduling activities for Day {day + 1}:")
        # Alternate between specific and general activities
        for i in range(activities_per_day):
            if i % 2 == 0 and specific_activities:
                activity = specific_activities.pop(0)
                today_activities.append(activity)
                print(f"Added specific activity: {activity['name']} with ratings {activity.get('rating', 'No rating')}")
            elif general_activities:
                activity = general_activities.pop(0)
                today_activities.append(activity)
                print(f"Added general activity: {activity['name']} with ratings {activity.get('rating', 'No rating')}")

        # Fill in remaining slots if necessary
        while len(today_activities) < activities_per_day and general_activities:
            activity = general_activities.pop(0)
            today_activities.append(activity)
            print(f"Added general activity to fill: {activity['name']} with ratings {activity.get('rating', 'No rating')}")

        print(f"Final activities for Day {day + 1}: {[activity['name'] for activity in today_activities]}")
        activity_selection.extend(today_activities)

    return activity_selection


def get_meal_options(food_preferences, location, num_days):
    meal_options = {"breakfast": [], "lunch": [], "dinner": []}

    for meal_type in ["breakfast", "lunch", "dinner"]:
        all_places = []  # Collect all places for all preferences for each meal
        query = f"{meal_type} place"
        print(f"Restaurant {meal_type} day {num_days}")
        places = get_yelp_data(query, location)  # Adjust limit as needed
        if places:
            print(f"Found {len(places)} places for query: {query}")
            all_places.extend(places)
        else:
            print(f"No places found for query: {query}")
        if food_preferences:
            # Loop through each food preference and make separate requests,
            # explicitly appending "restaurant" to each preference to form the query
            for preference in food_preferences:
                query = f"{preference} {meal_type} place"
                print(f"{preference} restaurant {meal_type} day {num_days}")
                print(f"Fetching data for: {query}, Location: {location}")
                places = get_yelp_data(query, location, limit=5)  # Adjust limit as needed
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
    try:
        result = geocoder.geocode(location, countrycode='gb')  # Limit search to Great Britain
        for res in result:
            if 'components' in res and 'country' in res['components']:
                if res['components']['country'] == 'United Kingdom':
                    return True
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
    return False

def gpt_response(user_content, token = 399):
    system_content = "You are Lee, a versatile travel chatbot capable of engaging in general conversations and providing helpful responses to users about questions and itineraries on travel destinations in the UK. "
    system_content += "Remember, you can only advice the user on things that are related to places within the UK. Anything outside you cannot accept."
    system_message = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": user_content}
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[system_message, user_message],
            temperature=0.6,
            max_tokens=token
        )
        chat_response = response.choices[0].message.content
        return chat_response
    except Exception as e:
        return f"Sorry, I encountered an issue: {str(e)}"


def chat_gpt(messages):
    user_input = messages
    location, intent = extract_information(user_input)
    print(intent)
    if intent == 'plan_trip':
        # Proceed with location-specific logic for travel-related inquiries
        if location:
            if is_location_in_uk(location):
                location += ", UK"
        else:
            user_content = "Tell the user that they need to include a destination within the UK and provide popular places"
            return gpt_response(user_content)

            
        num_days = travel_plan_state.num_days

        print(f"{location}\n{travel_plan_state.food_preferences}\n{travel_plan_state.activity_preferences}\n{travel_plan_state.hobby_preferences}\n{num_days}\n")

        meal_options = get_meal_options(travel_plan_state.food_preferences, location, num_days)
        all_activities = get_activity_options(travel_plan_state.activity_preferences, location, num_days)

        if len(all_activities) < 2 * num_days:
            user_content = f"Tell the user that there wasn't enough activities for their trip. After telling them this, see if you can answer their input: {user_input}"
            return gpt_response(user_content)

        hotels = get_hotel_data(location, all_activities, meal_options["breakfast"] + meal_options["lunch"] + meal_options["dinner"])
        travel_plan_state.update_plan(activity_options=all_activities, meal_options=meal_options, hotel_options=hotels)
        

        # Start with a clear and concise introduction for the itinerary
        prompt_summary = f"Create a concise travel itinerary for {num_days}-day trip to {location}, including dining, activities, and hotel recommendations.\n"
        # Add hobbies and interests in a summarized form
        if travel_plan_state.hobby_preferences:
            prompt_summary += f"Hobbies/Interests: {', '.join(travel_plan_state.hobby_preferences)}\n"
        # Summarize meal options
        if meal_options:
            prompt_summary += "Meal options:\n"
            for meal_type, meals in meal_options.items():
                meal_descriptions = [
                    f"{meal['name']} with a rating of {meal.get('rating', 'No rating')} stars and open from {meal.get('opening_times', 'Not available')}" 
                    for meal in meals
                ]
                prompt_summary += f"- {meal_type.capitalize()}: " + ", ".join(meal_descriptions) + "\n"

        # Summarize activities
        if all_activities:
            activity_descriptions = [
                f"{activity['name']} with a rating of {activity.get('rating', 'No rating')} stars and open from {activity.get('opening_times', 'Not available')}" 
                for activity in all_activities
            ]
            prompt_summary += "Activities: " + ", ".join(activity_descriptions) + "\n"

        # Add hotel recommendations
        if hotels:
            prompt_summary += "Hotel recommendations: " + ", ".join([hotel['hotel']['name'] for hotel in hotels[:5]]) + "\n"  # Limit to 5 examples
            prompt_summary += "Give options to users at the end and don't be putting users in different hotels for different days.\n"

        # Include the original user input for context
        prompt_summary += "User's request: " + user_input + "\n"

        # Sign off with a directive
        prompt_summary += "Take a look at the user's request and organize the informataion into a detailed, yet concise travel plan. Find different ways to include ratings for the busiensses."
        print(prompt_summary)

        return gpt_response(prompt_summary)
    else:
        return gpt_response(user_input, token=250)

# user_input = "I want to go to Lake District for a day."
# doc = nlp(user_input)
# print("Recognized entities and their labels:")
# for ent in doc.ents:
#     print(f"{ent.text} ({ent.label_})")
# # Extract location
# location = None
# for ent in doc.ents:
#     if ent.label_ == "GPE":  # GPE stands for geopolitical entity
#         location = ent.text
#         if location == "uk":
#             location = None
#         break

# if location:
#     print(f"Extracted location: {location}")
# else:
#     print("No suitable location found.")
# location += ", UK"
# is_location_in_uk(location)