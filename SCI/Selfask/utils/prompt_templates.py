
def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Self-Ask prompting strategy to analyze claims. Your task is to classify each claim into one of three categories — SUPPORTS, REFUTES, or NOINFO — based solely on the provided evidence.

Here are examples:
Example 1:  
Claim: Antibiotic induced alterations in the gut microbiome reduce resistance against Clostridium difficile.
Evidence: [0]Antibiotics can have significant and long-lasting effects on the gastrointestinal tract microbiota, reducing colonization resistance against pathogens including Clostridium difficile. [4]Our results indicate that antibiotic-mediated alteration of the gut microbiome converts the global metabolic profile to one that favours C. difficile germination and growth.
Are follow-up questions needed here: Yes
Follow-up 1: Do the evidences indicate that antibiotics alter the gut microbiome?
Intermediate answer 1: [0] states antibiotics have significant, long-lasting effects on the GI microbiota, and [4] explicitly mentions antibiotic-mediated alteration of the gut microbiome.
Follow-up 2: Do the evidences connect these alterations to reduced colonization resistance against C. difficile?
Intermediate answer 2: [0] explicitly says the alterations reduce colonization resistance against pathogens including C. difficile.
Follow-up 3: Do the evidences provide a mechanism consistent with decreased resistance to C. difficile?
Intermediate answer 3: [4] reports the altered microbiome shifts the global metabolic profile to favour C. difficile germination and growth, which aligns with reduced resistance.
Final label: SUPPORTS

Example 2:  
Claim: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs.
Evidence: [5]Pharmacological activation of AMPK in myofibroblasts from lungs of humans with IPF display lower fibrotic activity, along with enhanced mitochondrial biogenesis and normalization of sensitivity to apoptosis. [6]In a bleomycin model of lung fibrosis in mice, metformin therapeutically accelerates the resolution of well-established fibrosis in an AMPK-dependent manner. [7]These studies implicate deficient AMPK activation in non-resolving, pathologic fibrotic processes, and support a role for metformin (or other AMPK activators) to reverse established fibrosis by facilitating deactivation and apoptosis of myofibroblasts.
Are follow-up questions needed here: Yes
Follow-up 1: Does the evidence say that AMPK activation increases fibrosis in the lungs?
Intermediate answer 1: [5] shows the opposite: AMPK activation lowers fibrotic activity.
Follow-up 2: Does the evidence describe the effect of AMPK activation on fibrosis resolution?
Intermediate answer 2: [6] shows that in mice, metformin works in an AMPK-dependent way to speed up the resolution of fibrosis.
Follow-up 3: Do the studies link low AMPK activation to more severe fibrosis?
Intermediate answer 3: [7] says deficient AMPK activation is connected to non-resolving fibrosis, while activation helps reverse it.
Final label: REFUTES 

Example 3:  
Claim: 0-dimensional biomaterials show inductive properties.
Evidence: [No supporting evidence provided]
Are follow-up questions needed here: No  
Final label: NOINFO  

Now consider the following case:
Question: {claim} 
Evidence: {evidence} 

Are follow up questions needed here: Yes/No 
[If YES] Follow up:
Intermediate answer: [Answer based on the evidence]
[Optional if still undecidable]Follow up:
Intermediate answer: [Answer based on the evidence]
[Optional if still undecidable]Follow up:
Intermediate answer: [Answer based on the evidence]
... 
So the final answer is: one of SUPPORTS, REFUTES, or NOINFO.
"""
