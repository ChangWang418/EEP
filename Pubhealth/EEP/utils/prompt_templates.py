def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent trained to interrogate claims by simulating possible truths and contradictions.

Your task is to classify each claim into one of four categories: TRUE, FALSE, MIXTURE, or UNPROVEN, based on the provided evidence.

Please see the following examples:

Example 1:  
Claim: "Bats in Sierra Leone have tested positive for the Ebola virus."
Evidence: "Bats in Sierra Leone tested positive for the Ebola virus in recent surveillance studies."
Step 1: If the claim is true, I’d expect to see scientific findings or surveillance data reporting Ebola in local bats.
Step 2: The evidence directly states that such studies have confirmed it.
Step 3: I don’t see anything here that contradicts or casts doubt on the claim.
Step 4: The information seems clear, recent, and based on actual testing.
Answer: TRUE

Example 2:  
Claim: "Germany has banned the consumption of pork."
Evidence: "There is no official ban on pork in Germany; this claim originated from a satirical article."
Step 1: If this were true, I’d expect to find news reports or government policy confirming such a ban.
Step 2: Instead, the evidence says the claim came from a satire piece and that no such ban exists.
Step 3: This is a direct contradiction to the claim.
Step 4: The source is credible and clearly debunks the claim.
Answer: FALSE

Example 3:  
Claim: "Merck’s painkiller is effective and safe for long-term use."
Evidence: "The drug showed moderate efficacy but also raised cardiovascular concerns in long-term trials."
Step 1: If the claim is entirely accurate, I’d expect evidence of both strong efficacy and safety over time.
Step 2: The drug seems to work, but the safety data raises red flags.
Step 3: So, it partially supports the claim but also undermines it on a key point.
Step 4: The trial data is legitimate, but the conclusion is mixed.
Answer: MIXTURE

Example 4:  
Claim: "A new vaccine caused the deaths of 25 infants."
Evidence: "No scientific studies confirm a causal link between the vaccine and infant deaths."
Step 1: If this were true, I’d expect strong epidemiological studies or official investigations confirming causality.
Step 2: However, the evidence says no such studies exist.
Step 3: That doesn’t mean the claim is false — just that there’s no solid confirmation.
Step 4: The evidence is weak or inconclusive; I can’t confidently verify the claim.
Answer: UNPROVEN

Now, follow the same thinking steps **in your mind**, and output only the final classification label below.

Claim: "{claim}"  
Evidence: "{evidence}"  

You must **think through the following 4-step reasoning process internally**, but only output the final classification label:

Step 1: If the claim were true, what would strong evidence look like?  
Step 2: Does the actual evidence clearly support all parts of the claim?  
Step 3: If the evidence only partially supports the claim, is there also information that contradicts or undermines it?  
Step 4: If the evidence is vague, lacks verification, or comes from weak sources (e.g. social media), is the claim still verifiable?  
Step 5: Based on this, is the claim:
- TRUE (clearly supported by evidence)
- FALSE (clearly contradicted by evidence)
- MIXTURE (partly supported and partly contradicted)
- UNPROVEN (unclear, unverifiable, or not enough evidence)

Answer (TRUE / FALSE / MIXTURE / UNPROVEN):
"""
