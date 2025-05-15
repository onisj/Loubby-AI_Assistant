from groq import Groq
import os
import argostranslate.package
import argostranslate.translate

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(query, retrieved_docs, language="en-US"):
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext: {context}\nAnswer concisely:"
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        text = response.choices[0].message.content
        print("Groq response:", text)  # Debug
        if language != "en-US":
            from_code = "en"
            to_code = language.split("-")[0]
            print(f"Translating to {to_code}")  # Debug
            translated = argostranslate.translate.translate(text, from_code, to_code)
            print("Translated:", translated)  # Debug
            return translated
        return text
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")  # Debug
        return f"Error generating response: {str(e)}"