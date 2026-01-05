def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Standard Prompting (answer only) strategy to analyze claims. Your task is to classify each claim into one of three categories — SUPPORTS, REFUTES, or NOT ENOUGH INFO — based solely on the provided evidence.

Here are examples:
Example 1:  
Claim: L.A. Reid has served as the president of a record label.
Evidence: He has served as the chairman and CEO of Epic Records, a division of Sony Music Entertainment, the president and CEO of Arista Records, and the chairman and CEO of the Island Def Jam Music Group. 
Final label: SUPPORTS

Example 2:  
Claim: Mogadishu is located in Italy. 
Evidence: Mogadishu, known locally as Hamar, is the capital and most populous city of Somalia. 
Final label:  REFUTES

Example 3:  
Claim: Tilda Swinton is a vegan.
Evidence: [No supporting evidence provided]
Final label:  NOT ENOUGH INFO

Now consider the following case:
Claim: "{claim}"  
Evidence: "{evidence}"  

Final label: one of SUPPORTS, REFUTES, or NOT ENOUGH INFO.
"""
