
import os
import anthropic

client_claude = anthropic.Anthropic(api_key='k-ant-api03-R4QkkttqUInm29j1wxvthIm8ScvSrGZLX1nu1-9vfi2hme7g0AgrALYR9fizyj-Zo0kmvMK8PLyaf7nGInXPGA-IqhHiAAA')

def data_summary(df,check_name):
    data_json = df.to_json(orient='records')

    prompt = f"""
        You are an AI data analyst. Analyze the following data quality check results provided in JSON format:
        
        **Data:** {data_json}
        
        Your task is to:
        1. Evaluate the results of the data quality check.
        2. Identify key issues, patterns, or anomalies in the data.
        3. Generate a summary in **exactly 5 bullet points**.
        
        Instructions:
        - Be **concise, precise, and to the point**.
        - Do **not** add any assumptions, explanations, or extra text beyond the key findings.
        - Focus only on what the data shows.
        
        Output format:
        - 5 bullet points summarizing the data quality check findings.
        """
        
    response = client_claude.messages.create(
            model='claude-3-5-sonnet-20241022',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=1024
        )

    return response.content[0].text

  
