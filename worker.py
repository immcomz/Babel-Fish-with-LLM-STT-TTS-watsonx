import requests

# IBM Watson Machine Learning imports for using watsonx.ai foundation models
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods


# -----------------------------
# Watsonx.ai Model Configuration
# -----------------------------

# If running outside Skills Network, use your own API key and project ID
# API_KEY = "YOUR_WATSONX_API_KEY"
PROJECT_ID = "skills-network"

# IBM watsonx.ai service credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    # "apikey": API_KEY
}

# Model used for inference
model_id = "mistralai/mistral-medium-2505"

# Model generation parameters
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024,
}

# Initialize watsonx.ai LLM model
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID,
)


# -----------------------------
# Speech-to-Text Function
# -----------------------------

def speech_to_text(audio_binary):
    """
    Converts audio binary data into text using IBM Watson Speech-to-Text API.
    """

    # Watson Speech-to-Text base URL
    base_url = "..."

    # Full Speech-to-Text API endpoint
    api_url = base_url + "/speech-to-text/api/v1/recognize"

    # Speech recognition model
    params = {
        "model": "en-US_Multimedia",
    }

    # Send audio binary data to Watson Speech-to-Text service
    response = requests.post(
        api_url,
        params=params,
        data=audio_binary
    ).json()

    # Extract transcript from response
    text = "null"

    while bool(response.get("results")):
        print("Speech-to-Text response:", response)

        text = (
            response
            .get("results")
            .pop()
            .get("alternatives")
            .pop()
            .get("transcript")
        )

        print("Recognized text:", text)
        return text

    return text


# -----------------------------
# Text-to-Speech Function
# -----------------------------

def text_to_speech(text, voice=""):
    """
    Converts text into speech audio using IBM Watson Text-to-Speech API.
    """

    # Watson Text-to-Speech base URL
    base_url = "..."

    # Full Text-to-Speech API endpoint
    api_url = base_url + "/text-to-speech/api/v1/synthesize?output=output_text.wav"

    # Add selected voice if provided
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # HTTP headers for audio response
    headers = {
        "Accept": "audio/wav",
        "Content-Type": "application/json",
    }

    # Text payload sent to Watson Text-to-Speech
    json_data = {
        "text": text,
    }

    # Send request to Watson Text-to-Speech service
    response = requests.post(
        api_url,
        headers=headers,
        json=json_data
    )

    print("Text-to-Speech response:", response)

    # Return generated audio content
    return response.content


# -----------------------------
# Watsonx.ai Message Processing
# -----------------------------

def watsonx_process_message(user_message):
    """
    Sends the user's message to the watsonx.ai LLM and returns the response.
    """

    # Prompt sent to the model
    prompt = f"""
Respond clearly and helpfully to the following user query:

User query:
```{user_message}```
"""

    # Generate response from watsonx.ai model
    response_text = model.generate_text(prompt=prompt)

    print("Watsonx response:", response_text)

    # Return clean response text
    return response_text.strip()