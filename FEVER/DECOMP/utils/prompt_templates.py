def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Decompose-Then-Reason (DECOMP) strategy to analyze claims. Your task is to classify each claim into one of three categories — SUPPORTS, REFUTES, or NOT ENOUGH INFO — based solely on the provided evidence.

Here are some examples:
Example 1:
Claim: L.A. Reid has served as the president of a record label.
Evidence: He has served as the chairman and CEO of Epic Records, a division of Sony Music Entertainment, the president and CEO of Arista Records, and the chairman and CEO of the Island Def Jam Music Group.
QC: L.A. Reid has served as the president of a record label.
QS [qa]: Does the evidence mention that L.A. Reid held the role of president at a record label?
A: The evidence clearly states that he was the president and CEO of Arista Records, which is a record label.
QS [final_decision]: What is the final label of the claim?
A: SUPPORTS
QS [EOQ]

Example 2:
Claim: Mogadishu is located in Italy.
Evidence: Mogadishu, known locally as Hamar, is the capital and most populous city of Somalia.
QC: Mogadishu is located in Italy.
QS [qa]: According to the evidence, where is Mogadishu located?
A: The evidence says Mogadishu is the capital and most populous city of Somalia.
QS [qa]: Is this location (“Somalia”) consistent with the claim (“Italy”)?
A: The evidence places Mogadishu in Somalia, which contradicts the claim that it is in Italy.
QS [final_decision]: What is the final label of the claim?
A: REFUTES
QS [EOQ]

Example 3:
Claim: "Tilda Swinton is a vegan."
Evidence: "[No supporting evidence provided]"
QC: Tilda Swinton is a vegan.
QS: [qa] Does the evidence state that Tilda Swinton is a vegan?
A: The evidence does not mention this.
QS: [qa] Does the evidence provide any contrary statement?
A: There is neither support nor contradiction.
QS: [final_decision] What is the final label of the claim?
A: NOT ENOUGH INFO
QS: [EOQ]

Now consider the following case:
Claim: "{claim}"
Evidence: "{evidence}"

QC: {claim}
QS: [qa] Question 1 (derived from QC, only if needed)
A: Answer 1 (extracted from Evidence)
QS: [qa] Question 2 (derived from QC, only if needed)
A: Answer 2 (extracted from Evidence)
...
QS: [final_decision] Based on all the above, what is the final label of the claim?
A:  one of SUPPORTS, REFUTES, or NOT ENOUGH INFO.
QS: [EOQ]
"""