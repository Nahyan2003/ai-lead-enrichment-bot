import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

def get_company_summary(company_name):
    prompt = f"""
Please ONLY respond with a valid JSON object with EXACTLY these keys:

{{
  "name": "",
  "industry": "",
  "headquarters": "",
  "founded": "",
  "employee_count": "",
  "key_products_or_services": "",
  "summary": "",
  "website": "",
  "ai_automation_idea": ""
}}

Provide realistic answers for the company named "{company_name}".
Make the summary detailed (2-3 sentences).
If any info is unknown, use "unknown".
DO NOT add anything outside this JSON.
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()

        json_start = reply.find("{")
        json_end = reply.rfind("}") + 1

        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON object found in response.")

        json_string = reply[json_start:json_end]
        return json.loads(json_string)

    except Exception as e:
        return {
            "error": f"❌ Failed to parse response: {e}",
            "raw_response": reply if 'reply' in locals() else '(empty or no response)'
        }

def display_company_info(data):
    if "error" in data:
        st.error(data["error"])
        with st.expander("🔍 Raw API Response"):
            st.text(data.get("raw_response", "No raw response"))
        return

    st.markdown(f"### {data.get('name', 'Unknown')}")
    st.markdown(f"**Industry:** {data.get('industry', 'Unknown')}")
    st.markdown(f"**Headquarters:** {data.get('headquarters', 'Unknown')}")
    st.markdown(f"**Founded:** {data.get('founded', 'Unknown')}")
    st.markdown(f"**Employees:** {data.get('employee_count', 'Unknown')}")
    st.markdown(f"**Website:** {data.get('website', 'Unknown')}")
    st.markdown(f"**Key Products/Services:** {data.get('key_products_or_services', 'Unknown')}")
    st.markdown("**Summary:**")
    st.write(data.get("summary", "Not available"))
    st.markdown("**AI Automation Idea:**")
    st.write(data.get("ai_automation_idea", "Not available"))

# UI
st.title("🤖 AI Company Enrichment (Groq + LLaMA3)")
company_name = st.text_input("Enter company name:")
if st.button("Generate Company Info") and company_name.strip():
    with st.spinner("Fetching company details..."):
        result = get_company_summary(company_name.strip())
    display_company_info(result)
