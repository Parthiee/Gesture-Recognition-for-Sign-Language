import requests

try:
    url = " https://truly-keen-coyote.ngrok-free.app/start_processing"  # Replace with the actual URL of your API endpoint
    response = requests.post(url)
    # Check the response if needed
    if response.status_code != 200:
        print("Error:", response.text)
except Exception as e:
    print("Error making request:", e)
