def build_self_ask_prompt(claim, evidence):
    return f"""

You are a fact-checking agent trained to investigate claims by asking yourself follow-up questions and reasoning through intermediate steps.

Please respond with **only one** of the following labels: TRUE, FALSE, MIXTURE, or UNPROVEN.

Question: Bats in Sierra Leone have tested positive for the Ebola virus.
Evidence: "Bats in Sierra Leone tested positive for the Ebola virus in recent surveillance studies."
Are follow-up questions needed here: Yes.
Follow-up: Have scientific studies confirmed Ebola virus presence in Sierra Leone bats?
Intermediate answer: Yes, the evidence says recent surveillance studies detected Ebola in bats.
So the final answer is: TRUE

Question: Germany has banned the consumption of pork.
Evidence: "There is no official ban on pork in Germany; this claim originated from a satirical article."
Are follow-up questions needed here: Yes.
Follow-up: Is there any government or legal action that confirms a pork ban in Germany?
Intermediate answer: No, the evidence directly refutes the claim and traces it to satire.
So the final answer is: FALSE

Question: Merck’s painkiller is effective and safe for long-term use.
Evidence: "The drug showed moderate efficacy but also raised cardiovascular concerns in long-term trials."
Are follow-up questions needed here: Yes.
Follow-up: Do the trials show both high efficacy and long-term safety?
Intermediate answer: The drug works moderately, but safety is questionable.
So the final answer is: MIXTURE

Question: A new vaccine caused the deaths of 25 infants.
Evidence: "No scientific studies confirm a causal link between the vaccine and infant deaths."
Are follow-up questions needed here: Yes.
Follow-up: Is there verified evidence proving causality between the vaccine and the deaths?
Intermediate answer: No, studies do not confirm causation.
So the final answer is: UNPROVEN

Question: {claim}
Evidence: "{evidence}"
Are follow-up questions needed here: Yes.
Follow-up:
Intermediate answer:
So the final answer is:"""
