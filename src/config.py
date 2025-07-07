import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Add more config variables as needed 