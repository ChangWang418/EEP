def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent trained to assess natural language claims by reasoning step-by-step.

Your task is to classify each claim into one of four categories: TRUE, FALSE, MIXTURE, or UNPROVEN, based on the provided evidence.

Please see the following examples:

Example 1:  
Claim: "Bats in Sierra Leone have tested positive for the Ebola virus."  
Evidence: "Bats in Sierra Leone tested positive for the Ebola virus in recent surveillance studies."  
Step 1: The claim asserts bats in Sierra Leone tested positive for Ebola.  
Step 2: The evidence directly confirms this result from studies.  
Step 3: The claim is fully supported by the evidence.  
Answer: TRUE

Example 2:  
Claim: "Germany has banned the consumption of pork."  
Evidence: "There is no official ban on pork in Germany; this claim originated from a satirical article."  
Step 1: The claim asserts a national pork ban in Germany.  
Step 2: The evidence says the ban does not exist and the story is satire.  
Step 3: The evidence clearly refutes the claim.  
Answer: FALSE

Example 3:  
Claim: "Merck’s painkiller is effective and safe for long-term use."  
Evidence: "The drug showed moderate efficacy but also raised cardiovascular concerns in long-term trials."  
Step 1: The claim says the drug is both effective and safe.  
Step 2: The evidence presents partial support and partial contradiction.  
Step 3: The evidence supports efficacy but raises safety issues.  
Answer: MIXTURE

Example 4:  
Claim: "A new vaccine caused the deaths of 25 infants."  
Evidence: "No scientific studies confirm a causal link between the vaccine and infant deaths."  
Step 1: The claim asserts the vaccine caused multiple deaths.  
Step 2: The evidence indicates a lack of scientific verification.  
Step 3: Without confirmation, the claim remains unproven.  
Answer: UNPROVEN

Now consider the following case:

Claim: "{claim}"  
Evidence: "{evidence}"  

You must **think through the following 4-step reasoning process internally**, but only output the final classification label:

Step 1: Identify the factual assertion made by the claim.  
Step 2: Extract the most relevant facts from the evidence.  
Step 3: Reason logically about whether the evidence supports, contradicts, partially supports, or fails to verify the claim.  
Step 4: Decide whether the claim is TRUE, FALSE, MIXTURE, or UNPROVEN.

Answer:
"""
