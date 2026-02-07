from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()

while True:
    user = input("You: ")
    if user.lower() in ["exit", "quit"]:
        break

    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": user}],
        timeout=10
    )
    print("AI:", resp.choices[0].message.content)
