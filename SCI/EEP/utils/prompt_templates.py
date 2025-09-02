def build_self_ask_prompt(claim, evidence):
    return f"""
Example 1:
Claim: Antibiotic induced alterations in the gut microbiome reduce resistance against Clostridium difficile.
Evidence: [0]Antibiotics can have significant and long-lasting effects on the gastrointestinal tract microbiota, reducing colonization resistance against pathogens including Clostridium difficile. [4]Our results indicate that antibiotic-mediated alteration of the gut microbiome converts the global metabolic profile to one that favours C. difficile germination and growth.
Step 1: If the claim were true, there should be evidence that antibiotics alter the gut microbiome in ways that reduce colonization resistance and make conditions favourable for C. difficile. 
Step 2: If the claim were false, there should be evidence that antibiotics do not reduce colonization resistance, or that such alterations do not involve C. difficile, or that antibiotic effects on the gut microbiome are unrelated to resistance against pathogens. 
Step 3: The actual evidence matches the expectation in Step 1. It explicitly states that antibiotics reduce colonization resistance against pathogens including C. difficile [0], and that antibiotic-induced alterations favour C. difficile germination and growth [4]. 
Step 4: Based on the above, the final label is: SUPPORTS

Example 2:
Claim: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs.
Evidence: [5]Pharmacological activation of AMPK in myofibroblasts from lungs of humans with IPF display lower fibrotic activity, along with enhanced mitochondrial biogenesis and normalization of sensitivity to apoptosis. [6]In a bleomycin model of lung fibrosis in mice, metformin therapeutically accelerates the resolution of well-established fibrosis in an AMPK-dependent manner. [7]These studies implicate deficient AMPK activation in non-resolving, pathologic fibrotic processes, and support a role for metformin (or other AMPK activators) to reverse established fibrosis by facilitating deactivation and apoptosis of myofibroblasts.
Step 1: If the claim were true, there should be evidence that AMPK activation increases fibrotic activity in the lungs or promotes inflammation-related fibrosis. 
Step 2: If the claim were false, there should be evidence that AMPK activation does not increase fibrosis, but instead reduces or resolves it; or that deficient AMPK activation, rather than activation, is associated with pathological fibrosis. 
Step 3: The actual evidence matches the expectation in Step 2. [5] shows that AMPK activation reduces fibrotic activity in human lung myofibroblasts. [6] demonstrates that metformin, acting in an AMPK-dependent manner, accelerates the resolution of fibrosis in mice. [7] indicates that deficient AMPK activation is linked to persistent pathological fibrosis, while AMPK activators help reverse it. 
Step 4: Based on the above, the final label is: REFUTES

Example 3:
Claim: 0-dimensional biomaterials show inductive properties.
Evidence: [No supporting evidence provided]
Step 1: If the claim were true, there should be evidence that inductive properties were observed or measured in 0D biomaterials.
Step 2: If the claim were false, there should be evidence that 0D biomaterials do not exhibit inductive properties, or that experimental results explicitly deny such properties under comparable conditions. 
Step 3: The provided evidence is absent/unrelated and matches neither Step 1 nor Step 2. 
Step 4: Based on the above, the final label is: NOINFO

Now consider the following case:
Claim: "{claim}"
Evidence: "{evidence}"

Step 1: If the claim were true, what evidence should be present?
Step 2: If the claim were false, what evidence should be present?
Step 3: Compare the actual evidence to these expectations — does it align more with Step 1(SUPPORTS), Step 2(REFUTES), or neither(NOINFO)?
Step 4: Based on the above, the final label is: one of SUPPORTS, REFUTES, or NOINFO.
"""
