import os
from groq import Groq

# Tenta pegar do ambiente
api_key = os.environ.get("GROQ_API_KEY")

# DIAGNÓSTICO: Isso vai te mostrar o que o Python está realmente lendo
if api_key:
    # Mostra apenas os 4 primeiros e 4 últimos caracteres por segurança
    print(f"Key detected: {api_key[:4]}...{api_key[-4:]}")
else:
    print("No Key was detected by Python!")

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say Hello in English!"}]
    )
    print("Sucess:", response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")