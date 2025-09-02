def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Expectation–Evidence Prompting (EEP) strategy to analyze claims. Your task is to classify each claim into one of four categories — TRUE, FALSE, MIXTURE, or UNPROVEN — based solely on the provided evidence.

Here are examples:
Example 1:
Claim: Bat from Shawnee County tests positive for rabies.
Evidence: Topeka television station KSNT reports that the bat was found in Shawnee County. The Shawnee County Health Department is urging residents to be aware of the signs and symptoms of rabies and the steps to take if exposed. Rabies is a fatal but preventable viral disease that is typically transmitted by raccoons, bats, skunks and foxes. Health officials those who suspect they’ve been exposed to the disease should seek immediate medical treatment. Once a person begins to exhibit signs of disease, survival is rare.
Step 1: If the claim is true, there should be evidence that a bat in Shawnee County tested positive for rabies, supported by media reports and health department advisories.
Step 2: If the claim is false, there should be evidence that the bat tested negative, or that the positive case involved a different animal, or that the rabid bat was not from Shawnee County.
Step 3: The actual evidence shows that KSNT and the county health department confirmed a rabid bat in Shawnee County, which matches Step 1 and not Step 2; therefore, the claim is true.
Step 4: Based on the above, the final label is: TRUE

Example 2:
Claim: A new Facebook feature enabling users to report problems by shaking their phones has caused some users to be inadvertently reported for abuse and suspended.
Evidence: With little-to-no fanfare, Facebook began rolling out a new ‘shake-to-report’ feature in their mobile app in late May 2019. Facebook confirmed all this in an email to Snopes.com, including that ‘shake to report’ is enabled by default. First, it doesn’t automatically send a report to Facebook. It provides a link to a report form requiring the user to explain the problem. That means a report can’t be sent accidentally. Second, it clearly says that the shake-to-report feature is to report ‘something isn’t working,’ not to report abusive posts or other terms-of-service violations. In summary, is the ‘shake-to-report’ feature real? Yes, and if it has been rolled out to your app, it’s automatically enabled. You can disable it (if you wish to) by shaking your phone to call up the screen and turning it off, or by doing the same via the app’s Help & Support menu. Is it possible to inadvertently send reports or suspend someone’s account by using the feature? No, that’s not how it works.
Step 1: If the claim is true, there should be evidence that the feature automatically triggers abuse reports and leads to account suspensions.
Step 2: If the claim is false, there should be evidence that the feature does not exist, or that it only functions as technical feedback unrelated to abuse reports or suspensions.
Step 3: The actual evidence shows the feature does exist but only opens a feedback form, without triggering abuse reports or suspensions, which matches Step 2 and not Step 1; therefore, the claim is false.
Step 4: Based on the above, the final label is: FALSE

Example 3:
Claim: Drug effective in smoking cessation studies
Evidence: The numbers were from 2 fairly well-designed randomized, double-blind trials (N=2000 total). Abstinence was evaluated between varenicline, bupropion and placebo groups at 12 weeks, then continuous abstinence for another 40 weeks. During weeks 9-52 carbon monoxide (CM) levels were taken to confirm quit rates. The numbers presented are the continuous abstinence rates from weeks 9-52, based on these CM measurements. Mentions other drug treatment, which was used as a comparison in this double-blind study; however, no quantitative comparison of non-drug methods of smoking cessation (e.g. behavioral interventions, support groups, nicotine replacement) with regards to abstinence rates.
Step 1: If the claim is true, there should be evidence that the drug achieved significantly higher quit rates than placebo or control groups in randomized controlled trials, and that its effectiveness was recognized.
Step 2: If the claim is false, there should be evidence that the drug showed no significant difference compared to placebo or control groups, that its effect was uncertain, or that it had notable adverse effects.
Step 3: The actual evidence shows the drug outperformed placebo in trials (matching Step 1), but lacked comparisons with non-drug methods (partly matching Step 2), so the conclusion is partial support.
Step 4: Based on the above, the final label is: MIXTURE

Example 4:
Claim: Patients should avoid taking ibuprofen to relieve pain and fever associated with COVID-19 infections.
Evidence: the French government, including Health Minister Olivier Véran, issued warnings advising that infected persons should avoid taking nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen. Serious adverse events related to the use of nonsteroidal anti-inflammatory drugs (NSAIDs) have been reported in patients with COVID19, possible or confirmed cases. COVID-19 — Taking anti-inflammatory drugs (ibuprofen, cortisone, …) could be a factor in worsening the infection. If we take medicines that dampen this immune response, such as ibuprofen, this can lead to us not fighting off the infection as effectively, potentially leading to a longer illness with a higher risk of complications. These warnings to avoid ibuprofen (commonly known by the brand name Advil) generated mixed reactions among the medical community, with some asserting that scientific evidence to support it was lacking, and others maintaining that it was generally good advice.
Step 1: If the claim is true, there should be evidence showing that ibuprofen indeed worsens COVID-19 outcomes, supported by clinical or immunological studies and endorsed in authoritative guidelines.
Step 2: If the claim is false, there should be evidence showing that ibuprofen does not worsen COVID-19, that its use makes no difference in illness duration or complication risk, and that this is confirmed by authoritative sources.
Step 3: The actual evidence consists mainly of government warnings and expert speculation, lacking conclusive research support (not meeting Step 1), and it also does not establish that ibuprofen is completely safe (not meeting Step 2); therefore, the conclusion is unproven.
Step 4: Based on the above, the final label is: UNPROVEN

Now consider the following case:
Claim: "{claim}"  
Evidence: "{evidence}"  

Step 1: If the claim were true, what evidence should be present?
Step 2: If the claim were false, what evidence should be present?
Step 3: Compare the actual evidence to these expectations — does it fully match Step 1 (TRUE), fully match Step 2 (FALSE), contain elements matching both Step 1 and Step 2 (MIXTURE), or match neither (UNPROVEN)?
Step 4: Based on the above, the final label is: one of TRUE, FALSE, UNPROVEN, or MIXTURE.
"""
