def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Chain-of-Thought (CoT) strategy to analyze claims. Your task is to classify each claim into one of three categories — SUPPORTS, REFUTES or NOINFO — based solely on the provided evidence.

Here are examples:
Example 1:
Claim: Antibiotic induced alterations in the gut microbiome reduce resistance against Clostridium difficile.
Evidence: [0]Antibiotics can have significant and long-lasting effects on the gastrointestinal tract microbiota, reducing colonization resistance against pathogens including Clostridium difficile. [4]Our results indicate that antibiotic-mediated alteration of the gut microbiome converts the global metabolic profile to one that favours C. difficile germination and growth.
Reasoning: The evidence directly reports that antibiotics alter the gut microbiota and reduce colonization resistance against pathogens including Clostridium difficile [0]. It further states that these alterations change the metabolic profile to one that favours C. difficile germination and growth [4]. Together, these findings are consistent with the claim that antibiotic-induced microbiome changes reduce resistance to C. difficile.
Final label: SUPPORTS

Example 2:
Claim: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs.
Evidence: [5]Pharmacological activation of AMPK in myofibroblasts from lungs of humans with IPF display lower fibrotic activity, along with enhanced mitochondrial biogenesis and normalization of sensitivity to apoptosis. [6]In a bleomycin model of lung fibrosis in mice, metformin therapeutically accelerates the resolution of well-established fibrosis in an AMPK-dependent manner. [7]These studies implicate deficient AMPK activation in non-resolving, pathologic fibrotic processes, and support a role for metformin (or other AMPK activators) to reverse established fibrosis by facilitating deactivation and apoptosis of myofibroblasts.
Reasoning: The evidence consistently shows that AMPK activation reduces, rather than increases, fibrotic activity. In human lung myofibroblasts, AMPK activation lowers fibrosis [5]. In a mouse model, AMPK-dependent activation by metformin accelerates fibrosis resolution [6]. Moreover, deficient AMPK activation is linked to persistent pathological fibrosis, and activators like metformin help reverse fibrosis [7]. Together, these results contradict the claim that AMPK activation increases lung fibrosis.
Final label: REFUTES

Example 3:
Claim: 0-dimensional biomaterials show inductive properties.
Evidence: [No supporting evidence provided]
Reasoning: The provided evidence is insufficient/unrelated, so it neither supports nor contradicts the claim.
Final label: NOINFO

Now consider the following case:
Claim: "{claim}"
Evidence: "{evidence}"

Reasoning:
Final label: one of SUPPORTS, REFUTES, or NOINFO.
"""
