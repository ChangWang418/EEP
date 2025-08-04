def build_self_ask_prompt(claim, evidence):
    return f"""

You are a fact-checking agent trained to interrogate claims by simulating possible truths and contradictions.

Here are examples:

Claim: "The Eiffel Tower is located in Berlin."  
Evidence: "The Eiffel Tower is in Paris."  
Step 1: If the claim were true, we would expect to find evidence confirming the Eiffel Tower is in Berlin.  
Step 2: The actual evidence says it is in Paris.  
Step 3: This directly contradicts the expected location.  
Step 4: Based on this, the claim is: REFUTES

Claim: "Barack Obama served as U.S. President."  
Evidence: "Barack Obama was the 44th President of the United States."  
Step 1: If the claim were true, we would expect evidence confirming Obama held presidential office.  
Step 2: The evidence confirms he was the 44th President.  
Step 3: This fully aligns with the claim.  
Step 4: Based on this, the claim is: SUPPORTS

Claim: "Tilda Swinton is a vegan."  
Evidence: "[No supporting evidence provided]"  
Step 1: If the claim were true, we would expect public statements, interviews, or verified sources confirming her dietary choices.  
Step 2: The evidence provides no such confirmation or any relevant information.  
Step 3: There is no contradictory evidence, but also no verification.  
Step 4: Based on this, the claim is: NOT ENOUGH INFO

Now consider the following case:

Claim: "{claim}"
Evidence: "{evidence}"

Step 1: If the claim were true, what evidence would you expect to find?
Step 2: Does the actual evidence align with that expectation?
Step 3: If not, does the evidence contradict it, or is it unrelated?
Step 4: Based on this, is the claim:
- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

Output only one label in uppercase. No explanation.
Answer:
"""
