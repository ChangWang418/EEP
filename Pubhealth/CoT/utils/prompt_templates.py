def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Chain-of-Thought (CoT) strategy to analyze claims. Your task is to classify each claim into one of four categories — TRUE, FALSE, MIXTURE, or UNPROVEN — based solely on the provided evidence.

Here are examples:
Example 1:
Claim: Bat from Shawnee County tests positive for rabies.
Evidence: Topeka television station KSNT reports that the bat was found in Shawnee County. The Shawnee County Health Department is urging residents to be aware of the signs and symptoms of rabies and the steps to take if exposed. Rabies is a fatal but preventable viral disease that is typically transmitted by raccoons, bats, skunks and foxes. Health officials those who suspect they’ve been exposed to the disease should seek immediate medical treatment. Once a person begins to exhibit signs of disease, survival is rare.
Reasoning: The evidence shows a bat in Shawnee County linked to rabies warnings, which supports the claim without contradiction.
Final label: TRUE

Example 2:
Claim: A new Facebook feature enabling users to report problems by shaking their phones has caused some users to be inadvertently reported for abuse and suspended.	
Evidence: With little-to-no fanfare, Facebook began rolling out a new ‘shake-to-report’ feature in their mobile app in late May 2019. Facebook confirmed all this in an email to Snopes.com, including that ‘shake to report’ is enabled by default. First, it doesn’t automatically send a report to Facebook. It provides a link to a report form requiring the user to explain the problem. That means a report can’t be sent accidentally. Second, it clearly says that the shake-to-report feature is to report ‘something isn’t working,’ not to report abusive posts or other terms-of-service violations. In summary, is the ‘shake-to-report’ feature real? Yes, and if it has been rolled out to your app, it’s automatically enabled. You can disable it (if you wish to) by shaking your phone to call up the screen and turning it off, or by doing the same via the app’s Help & Support menu. Is it possible to inadvertently send reports or suspend someone’s account by using the feature? No, that’s not how it works.
Reasoning: The evidence confirms the feature but directly contradicts the claim about accidental abuse reports and suspensions.
Final label: FALSE

Example 3:
Claim: Drug effective in smoking cessation studies	
Evidence: The numbers were from 2 fairly well-designed randomized, double-blind trials (N=2000 total). Abstinence was evaluated between varenicline, bupropion and placebo groups at 12 weeks, then continuous abstinence for another 40 weeks. During weeks 9-52 carbon monoxide (CM) levels were taken to confirm quit rates. The numbers presented are the continuous abstinence rates from weeks 9-52, based on these CM measurements.  Mentions other drug treatment, which was used as a comparison in this double-blind study; however, no quantitative comparison of non-drug methods of smoking cessation (e.g. behavioral interventions, support groups, nicotine replacement) with regards to abstinence rates.
Reasoning: The evidence supports drug effectiveness relative to other drugs and placebo but lacks comparison to non-drug methods, so support is partial.
Final label: MIXTURE

Example 4:
Claim: Patients should avoid taking ibuprofen to relieve pain and fever associated with COVID-19 infections.
Evidence: the French government, including Health Minister Olivier Véran, issued warnings advising that infected persons should avoid taking nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen. Serious adverse events related to the use of nonsteroidal anti-inflammatory drugs (NSAIDs) have been reported in patients with COVID19, possible or confirmed cases. COVID-19 — Taking anti-inflammatory drugs (ibuprofen, cortisone, …) could be a factor in worsening the infection. If we take medicines that dampen this immune response, such as ibuprofen, this can lead to us not fighting off the infection as effectively, potentially leading to a longer illness with a higher risk of complications. These warnings to avoid ibuprofen (commonly known by the brand name Advil) generated mixed reactions among the medical community, with some asserting that scientific evidence to support it was lacking, and others maintaining that it was generally good advice.
Reasoning: The warnings exist, but the evidence is not conclusive and lacks scientific consensus.
Final label: UNPROVEN

Now consider the following case:
Claim: "{claim}"
Evidence: "{evidence}"

Reasoning:
Final label: one of TRUE, FALSE, UNPROVEN, or MIXTURE.
"""
