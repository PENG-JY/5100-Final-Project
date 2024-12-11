# use openai api to determine whether a developer is famous
import openai
import json
import pandas as pd

# import openai api key
with open('/Users/rx/Documents/DSAN5000/Api_key/openai.json') as f:

    keys = json.load(f)

openai.api_key = keys['openai_api']

# define a function that uses chatgpt to analyze json file
def classify_developer(developer_name, model, temperature, max_tokens):

    prompt = f"""Classify the game developer '{developer_name}' into one of the following categories: Large Game Company, Indie Developer, Regional Developer.
                 If the game is developed by multiple developers, choose the most well-known or impactful one based on your judgment, then categorize it into one of the three categories mentioned above. 
                 Return only the classification result."""
    
    try:
        response = openai.ChatCompletion.create(
            model = model,  
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                ],
            temperature = temperature,
            max_tokens = max_tokens
        )
        
        reply = response['choices'][0]['message']['content'].strip()
        return reply
    
    except openai.OpenAIError as e:
        print(f"An error occurred {e}")
        return None

# import csv file and fetch developers
df = pd.read_csv("data/cleaned_game.csv")

# check NA
if df['developer'].isna().any():
    print("developer column has NA")
else:
    print("developer column doesn't have NA")

# change dtype
developers = df['developer']
developers = developers.tolist()

# use function
model="gpt-4o"
temperature=0.5
max_tokens=10

developer_category = []
for developer in developers:
    category = classify_developer(developer,model=model,temperature=temperature,max_tokens=max_tokens)
    developer_category.append(category)

df['developer_category'] = developer_category
df.to_csv('data/developer_category.csv', index=False)
