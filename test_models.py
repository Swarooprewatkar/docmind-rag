from google import genai

client = genai.Client(api_key="AIzaSyDl2w9j5XRU4jaMvUGARNuLv6viYHhiOMg")

for model in client.models.list():
    if "embed" in model.name.lower():
        print(model.name)
