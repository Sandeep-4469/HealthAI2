import openai
import json

openai.api_key = ""

def get_data():
    with open("static/data.json", "r") as json_file:
        data = json.load(json_file)
    return str(data)
print("..........................................")

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def collect_messages_text(msg,context): 
    print("..............collect.............")
    context.append({'role':'user', 'content':f"{msg}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    return response
