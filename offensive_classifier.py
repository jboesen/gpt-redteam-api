"""
Provides functionality to evaluate sentences based on Twitter Roberta Model
"""
import asyncio
import itertools
import json
from time import sleep
import httpx
import requests
import os

API_TOKEN = os.environ['HUGGINGFACE_API_TOKEN']
HEADERS = {'Authorization': 'Bearer {API_TOKEN}'}
REPO_ID = 'cardiffnlp/twitter-roberta-base-offensive'
API_URL = 'https://api-inference.huggingface.co/models/{REPO_ID}'

def offensive_query(payload):
    """Runs query to HuggingFace API"""
    response = requests.request("POST", API_URL, headers=HEADERS, data=json.dumps(payload))
    return json.loads(response.content.decode("utf-8"))

async def get_sentence_offensiveness(sentence, client):
    """Runs query for one sentence"""
    print('calling sentence_off')
    payload = { 'inputs': sentence }
    response = await client.post(
        f'https://api-inference.huggingface.co/models/{REPO_ID}', data=json.dumps(payload)
    )
    response_dict = response.json()
    # print(response_dict)
    if 'error' in response_dict:
        if response_dict['error'] == 'Rate limit reached. Please log in or use your apiToken':
            return response_dict
        sleep(100)
        get_sentence_offensiveness(sentence, client)
    # print(response.json())
    return response_dict

async def get_all_offensiveness(sentences):
    """Runs query for each sentence, returning probability of offensiveness"""
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *map(get_sentence_offensiveness, sentences, itertools.repeat(client))
        )
    for rating in results:
        if 'error' in rating\
             and rating['error'] == 'Rate limit reached. Please log in or use your apiToken':
            return [rating]
    formatted_response = []
    for i, rating in enumerate(results):
        formatted_response.append({'sentence': sentences[i], 'offensive': rating[0][1]['score']})
    return formatted_response

def sort_offensive(predictions_arr):
    """Rates and sorts offensiveness where key sentence and value is offensiveness"""
    offensive_vals = asyncio.run(get_all_offensiveness(predictions_arr))
    return sorted(offensive_vals, key=lambda d: d['offensive'], reverse=True)