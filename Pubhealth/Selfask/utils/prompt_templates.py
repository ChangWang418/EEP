def build_self_ask_prompt(claim, evidence):
    return f"""
You are a fact-checking agent. Use the Self-Ask prompting strategy to analyze claims. Your task is to classify each claim into one of four categories — TRUE, FALSE, MIXTURE, or UNPROVEN — based solely on the provided evidence.

Here are examples:
Example 1:
Claim: Bat from Shawnee County tests positive for rabies.
Evidence: Topeka television station KSNT reports that the bat was found in Shawnee County. The Shawnee County Health Department is urging residents to be aware of the signs and symptoms of rabies and the steps to take if exposed. Rabies is a fatal but preventable viral disease that is typically transmitted by raccoons, bats, skunks and foxes. Health officials those who suspect they’ve been exposed to the disease should seek immediate medical treatment. Once a person begins to exhibit signs of disease, survival is rare.
Are follow up questions needed here: Yes.
Follow up: Does the evidence describe whether the bat found in Shawnee County was confirmed rabid?
Intermediate answer: The evidence reports that the bat found in Shawnee County tested positive for rabies.
So the final answer is: TRUE

Example 2:
Claim: A new Facebook feature enabling users to report problems by shaking their phones has caused some users to be inadvertently reported for abuse and suspended.	
Evidence: With little-to-no fanfare, Facebook began rolling out a new ‘shake-to-report’ feature in their mobile app in late May 2019. Facebook confirmed all this in an email to Snopes.com, including that ‘shake to report’ is enabled by default. First, it doesn’t automatically send a report to Facebook. It provides a link to a report form requiring the user to explain the problem. That means a report can’t be sent accidentally. Second, it clearly says that the shake-to-report feature is to report ‘something isn’t working,’ not to report abusive posts or other terms-of-service violations. In summary, is the ‘shake-to-report’ feature real? Yes, and if it has been rolled out to your app, it’s automatically enabled. You can disable it (if you wish to) by shaking your phone to call up the screen and turning it off, or by doing the same via the app’s Help & Support menu. Is it possible to inadvertently send reports or suspend someone’s account by using the feature? No, that’s not how it works.
Are follow up questions needed here: Yes.
Follow up: How does the evidence describe the existence of this feature?
Intermediate answer: The evidence says the “shake-to-report” feature is real and is automatically enabled in the app.
Follow up: How does the evidence describe its relation to abuse reports or account suspensions?
Intermediate answer: The evidence explains it only opens a feedback form for technical problems and cannot trigger abuse reports or suspensions.
So the final answer is: FALSE

Example 3:
Claim: Drug effective in smoking cessation studies	
Evidence: The numbers were from 2 fairly well-designed randomized, double-blind trials (N=2000 total). Abstinence was evaluated between varenicline, bupropion and placebo groups at 12 weeks, then continuous abstinence for another 40 weeks. During weeks 9-52 carbon monoxide (CM) levels were taken to confirm quit rates. The numbers presented are the continuous abstinence rates from weeks 9-52, based on these CM measurements.  Mentions other drug treatment, which was used as a comparison in this double-blind study; however, no quantitative comparison of non-drug methods of smoking cessation (e.g. behavioral interventions, support groups, nicotine replacement) with regards to abstinence rates.
Are follow up questions needed here: Yes.
Follow up: How does the evidence indicate the drug’s effectiveness?
Intermediate answer: The evidence shows higher continuous abstinence rates in the drug groups compared with placebo, indicating effectiveness.
Follow up: Does the evidence also provide comparisons with non-drug interventions?
Intermediate answer: The evidence states that no quantitative comparison with non-drug methods was included.
So the final answer is: MIXTURE

Example 4:
Claim: Patients should avoid taking ibuprofen to relieve pain and fever associated with COVID-19 infections.
Evidence: the French government, including Health Minister Olivier Véran, issued warnings advising that infected persons should avoid taking nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen. Serious adverse events related to the use of nonsteroidal anti-inflammatory drugs (NSAIDs) have been reported in patients with COVID19, possible or confirmed cases. COVID-19 — Taking anti-inflammatory drugs (ibuprofen, cortisone, …) could be a factor in worsening the infection. If we take medicines that dampen this immune response, such as ibuprofen, this can lead to us not fighting off the infection as effectively, potentially leading to a longer illness with a higher risk of complications. These warnings to avoid ibuprofen (commonly known by the brand name Advil) generated mixed reactions among the medical community, with some asserting that scientific evidence to support it was lacking, and others maintaining that it was generally good advice.
Are follow up questions needed here: Yes.
Follow up: What positions do the government and some doctors take in the evidence?
Intermediate answer: The French government and some doctors advised against ibuprofen, expressing concern it could worsen illness.
Follow up: What does the evidence say about the scientific consensus?
Intermediate answer: The evidence shows the medical community was divided, with some experts saying there was no solid evidence to support the warning.
So the final answer is: UNPROVEN

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
So the final answer is: one of TRUE, FALSE, UNPROVEN, or MIXTURE.
"""
