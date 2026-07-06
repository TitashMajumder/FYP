import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(lat, lon):

     if not lat or not lon:
          return None
     try:
          url = (
               "https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}"
               f"&lon={lon}"
               f"&appid={API_KEY}"
               "&units=metric"
          )
          r = requests.get(url, timeout=10)
          data = r.json()
          return {
               "temperature": data["main"]["temp"],
               "humidity": data["main"]["humidity"],
               "pressure": data["main"]["pressure"],
               "wind_speed": data["wind"]["speed"],
               "weather": data["weather"][0]["main"],
               "description": data["weather"][0]["description"],
               "city": data["name"]
          }
     except Exception as e:
          print(f"Weather fetch failed: {e}")
          return None