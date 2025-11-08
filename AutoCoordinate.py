from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def _get_exif_data(image_path):
     """Internal helper to extract GPSInfo from image."""
     try:
          image = Image.open(image_path)
          exif_data_raw = image._getexif()
          
          if not exif_data_raw:
               return None
          
          gps_info = {}
          # Find the GPSInfo tag
          for tag, value in exif_data_raw.items():
               tag_name = TAGS.get(tag, tag)
               if tag_name == "GPSInfo":
                    # Decode the GPS data
                    for key in value.keys():
                         sub_tag = GPSTAGS.get(key, key)
                         gps_info[sub_tag] = value[key]
                    return gps_info # Return only the GPS data
                    
     except Exception as e:
          print(f"Error reading EXIF data: {e}")
          return None
     return None

def _convert_to_decimal(value):
     """Internal helper to convert GPS tuples to decimal."""
     try:
          # EXIF stores GPS as ((Degrees, 1), (Minutes, 1), (Seconds, 1))
          d = float(value[0][0]) / float(value[0][1])
          m = float(value[1][0]) / float(value[1][1])
          s = float(value[2][0]) / float(value[2][1])
          
          return d + (m / 60.0) + (s / 3600.0)
     except Exception:
          # Handle potential malformed GPS data (e.g., just a float)
          return float(value)

def get_lat_lon(image_path):
     """
     The main function.
     Extracts latitude and longitude from image EXIF metadata.
     Returns (lat, lon) as decimals or (None, None) if not found.
     """
     gps_info = _get_exif_data(image_path)
     
     if not gps_info:
          return None, None
          
     # Check if we have all the required tags
     if ('GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info and
          'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info):
          
          try:
               lat = _convert_to_decimal(gps_info['GPSLatitude'])
               lon = _convert_to_decimal(gps_info['GPSLongitude'])
               
               # Check North/South and East/West references
               if gps_info['GPSLatitudeRef'] == 'S':
                    lat = -lat
               if gps_info['GPSLongitudeRef'] == 'W':
                    lon = -lon
                    
               return lat, lon
               
          except Exception as e:
               print(f"Error converting GPS coordinates: {e}")
               return None, None
               
     return None, None