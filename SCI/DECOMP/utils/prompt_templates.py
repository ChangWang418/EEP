def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Decompose-Then-Reason (DECOMP) strategy to analyze claims. Your task is to classify each claim into one of three categories — SUPPORTS, REFUTES or NOINFO — based solely on the provided evidence.

Here are examples:
Example 1:
Claim: Antibiotic induced alterations in the gut microbiome reduce resistance against Clostridium difficile.
Evidence: [0]Antibiotics can have significant and long-lasting effects on the gastrointestinal tract microbiota, reducing colonization resistance against pathogens including Clostridium difficile. [4]Our results indicate that antibiotic-mediated alteration of the gut microbiome converts the global metabolic profile to one that favours C. difficile germination and growth.
QC: Antibiotics alter the gut microbiome and reduce resistance against Clostridium difficile.
QS: [qa] Does the evidence say antibiotics change the gut microbiome?
A: [0] says “Antibiotics can have significant and long-lasting effects on the gastrointestinal tract microbiota,” and [4] says “antibiotic-mediated alteration of the gut microbiome.”
QS: [qa] Does the evidence say these changes reduce resistance against Clostridium difficile?
A: [0] says antibiotics reduce “colonization resistance against pathogens including Clostridium difficile.”
QS: [qa] Does the evidence mention that the altered microbiome favours C. difficile growth?
A: [4] says the alteration “converts the global metabolic profile to one that favours C. difficile germination and growth.”
A: SUPPORTS
QS: [EOQ]

Example 2:
Claim: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs.
Evidence: [5]Pharmacological activation of AMPK in myofibroblasts from lungs of humans with IPF display lower fibrotic activity, along with enhanced mitochondrial biogenesis and normalization of sensitivity to apoptosis. [6]In a bleomycin model of lung fibrosis in mice, metformin therapeutically accelerates the resolution of well-established fibrosis in an AMPK-dependent manner. [7]These studies implicate deficient AMPK activation in non-resolving, pathologic fibrotic processes, and support a role for metformin (or other AMPK activators) to reverse established fibrosis by facilitating deactivation and apoptosis of myofibroblasts.
QC: AMPK activation increases inflammation-related fibrosis in the lungs.
QS: [qa] Does the evidence indicate that AMPK activation increases fibrosis?
A: [5] reports that AMPK activation in lung myofibroblasts from IPF patients shows lower fibrotic activity, not higher.
QS: [qa] Does the evidence indicate that AMPK activation helps resolve or reduce fibrosis?
A: [6] states that in a bleomycin lung fibrosis model, metformin (an AMPK activator) accelerates the resolution of established fibrosis in an AMPK-dependent way.
QS: [qa] Do the studies link deficient AMPK activation to worse fibrosis outcomes?
A: [7] says deficient AMPK activation is associated with non-resolving pathological fibrosis, and AMPK activators like metformin can help reverse fibrosis.
QS: [final_decision] What is the final label of the claim?
A: REFUTES
QS: [EOQ]

Example 3:
Claim: 0-dimensional biomaterials show inductive properties.
Evidence: [No supporting evidence provided]
QC: 0D biomaterials have inductive properties.
QS: [qa] Does the evidence mention inductive properties of 0D biomaterials?
A: It does not discuss inductive properties.
QS: [qa] Does the evidence provide an explicit contradiction?
A: It provides neither support nor contradiction.
QS: [final_decision] What is the final label of the claim?
A: NOINFO
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
A: one of SUPPORTS, REFUTES or NOINFO.
QS: [EOQ]
"""