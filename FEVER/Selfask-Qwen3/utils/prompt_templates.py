def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent trained to investigate claims by asking yourself follow-up questions and reasoning through intermediate steps.

Please respond with **only one** of the following labels: SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Example 1:  
Claim: The Eiffel Tower is located in Berlin.
Evidence: "The Eiffel Tower is in Paris."
Are follow-up questions needed here: Yes.
Follow-up: Where is the Eiffel Tower located?
Intermediate answer: The Eiffel Tower is located in Paris.
So the final answer is: REFUTES

Example 2:  
Claim: Barack Obama served as U.S. President.
Evidence: "Barack Obama was the 44th President of the United States."
Are follow-up questions needed here: Yes.
Follow-up: Did Barack Obama serve as President of the United States?
Intermediate answer: Yes, he was the 44th President.
So the final answer is: SUPPORTS

Example 3:  
Claim: Tilda Swinton is a vegan.
Evidence: "[No supporting evidence provided]"
Are follow-up questions needed here: Yes.
Follow-up: Is there any information about Tilda Swinton’s dietary habits?
Intermediate answer: The evidence does not mention her diet or lifestyle.
So the final answer is: NOT ENOUGH INFO

Now consider the following case:

Question: {claim}
Evidence: "{evidence}"

Are follow-up questions needed here: Yes.
Follow-up:
Intermediate answer:

So the final answer is:"""
