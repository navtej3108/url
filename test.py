from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import face_recognition
import aiohttp
import asyncio
from io import BytesIO

app = FastAPI()

class ImageComparisonRequest(BaseModel):
    profile_pics_url: str
    url1: str
    url2: str

async def fetch_image(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            else:
                return None
    except Exception as e:
        print(f"Error retrieving image from {url}: {e}")
        return None

async def get_face_encoding(image_data):
    if image_data:
        image = face_recognition.load_image_file(BytesIO(image_data))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return encodings[0]
    return None

async def compare_faces(image_encoding, comparison_url):
    async with aiohttp.ClientSession() as session:
        image_data = await fetch_image(session, comparison_url)
        
        if image_data is None:
            return False, 'URL not found'

        comparison_encoding = await get_face_encoding(image_data)
        
        if comparison_encoding is None:
            return False, 'No encoding found'
        
        matches = face_recognition.compare_faces([image_encoding], comparison_encoding, tolerance=0.4)
        if matches[0]:
            return True, 'Match'
        else:
            return False, 'Images not matching'

@app.post("/check-image-matches/")
async def check_image_matches(request: ImageComparisonRequest):
    profile_pics_url = request.profile_pics_url
    url1 = request.url1
    url2 = request.url2

    try:
        # Fetch the profile picture encoding
        async with aiohttp.ClientSession() as session:
            profile_image_data = await fetch_image(session, profile_pics_url)
            if profile_image_data is None:
                return {"status": "failure", "message": "Profile picture URL not found"}

            profile_encoding = await get_face_encoding(profile_image_data)
            if profile_encoding is None:
                return {"status": "failure", "message": "No encoding found for profile picture"}

        # Compare profile picture with url1
        match1, result1 = await compare_faces(profile_encoding, url1)
        
        # Compare profile picture with url2
        match2, result2 = await compare_faces(profile_encoding, url2)

        if match1 and match2:
            return {"status": "success", "message": "Both images match the profile picture"}
        elif match1:
            return {"status": "partial_match", "message": "Only url1 matches the profile picture"}
        elif match2:
            return {"status": "partial_match", "message": "Only url2 matches the profile picture"}
        else:
            return {"status": "failure", "message": "Neither of the images match the profile picture"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "API is working. Use /check-image-matches/ to compare images."}