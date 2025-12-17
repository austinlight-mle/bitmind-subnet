import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from the specified .env file
load_dotenv(dotenv_path='.env.gen_miner')
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("OPENAI_API_KEY not found in .env.gen_miner")
else:
    client = OpenAI(api_key=api_key)

    # Test image generation separately
    try:
        client.images.generate(
            model="gpt-image-1",
            prompt="A watercolor painting of a mountain landscape at sunrise.",
            size="1024x1024",
        )
        print("Image generation: VALID for this key")
    except Exception:
        print("Image generation: NOT VALID for this key")

    # Test video generation separately
    try:
        # Note: This initiates an asynchronous job. The actual video needs polling.
        # This call checks for initial authorization/model access.
        client.videos.create(
            model="sora-2",
            prompt="A test video of a flower blooming.",
            seconds="4",  # Sora expects '4', '8', or '12' as strings
            size="1024x1792",
        )
        print("Video generation: VALID for this key")
    except Exception:
        print("Video generation: NOT VALID for this key")

