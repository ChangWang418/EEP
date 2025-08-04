def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent trained to assess natural language claims by reasoning step-by-step.

Your task is to classify each claim into one of three categories: SUPPORTS, REFUTES, or NOT ENOUGH INFO, based on the provided evidence.

Please see the following examples:

Example 1:  
Claim: "The Eiffel Tower is located in Berlin."  
Evidence: "The Eiffel Tower is in Paris."  
Step 1: The claim asserts the Eiffel Tower is in Berlin.  
Step 2: The evidence states it is in Paris.  
Step 3: These locations contradict each other.  
Answer: REFUTES

Example 2:  
Claim: "Barack Obama served as U.S. President."  
Evidence: "Barack Obama was the 44th President of the United States."  
Step 1: The claim says Obama was a U.S. President.  
Step 2: The evidence confirms he was the 44th President.  
Step 3: This supports the claim directly.  
Answer: SUPPORTS

Example 3:  
Claim: "Tilda Swinton is a vegan."  
Evidence: "[No supporting evidence provided]"  
Step 1: The claim asserts that Tilda Swinton follows a vegan diet.  
Step 2: The evidence contains no information about her dietary habits.  
Step 3: Without relevant confirmation or contradiction, we cannot determine the truth of the claim.  
Answer: NOT ENOUGH INFO

Now, follow the same thinking steps **in your mind**, and output only the final classification label below.

Claim: "{claim}"  
Evidence: "{evidence}"  

You must **think through the following 4-step reasoning process internally**, but only output the final classification label:

Step 1: Identify the factual assertion made by the claim.  
Step 2: Extract the most relevant facts from the evidence.  
Step 3: Reason logically about whether the evidence supports or contradicts the claim.  
Step 4: Decide whether the claim is SUPPORTED, REFUTED, or whether there is NOT ENOUGH INFO.

Answer:
"""
