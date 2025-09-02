import os
import re
import pandas as pd
from model.gpt_verifier import gpt_self_ask_verifier

# ===== 你已有的提示词（原样使用）=====
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
[Optional if still undecidable] Follow up:
Intermediate answer: [Answer based on the evidence]
[Optional if still undecidable] Follow up:
Intermediate answer: [Answer based on the evidence]
... 
So the final answer is: one of TRUE, FALSE, UNPROVEN, or MIXTURE.
"""

# ===== 标签集合与别名 =====
VALID_LABELS = {"TRUE", "FALSE", "UNPROVEN", "MIXTURE"}
ALIAS_MAP = {
    # 英文别名
    "SUPPORT": "TRUE", "SUPPORTS": "TRUE", "SUPPORTED": "TRUE",
    "REFUTE": "FALSE", "REFUTES": "FALSE", "REFUTED": "FALSE",
    "YES": "TRUE", "NO": "FALSE",
    "MIXED": "MIXTURE", "PARTIAL SUPPORT": "MIXTURE", "PARTLY TRUE": "MIXTURE",
    "NOT ENOUGH EVIDENCE": "UNPROVEN", "INSUFFICIENT EVIDENCE": "UNPROVEN",
    "CANNOT BE DETERMINED": "UNPROVEN", "UNDETERMINED": "UNPROVEN",
    "UNKNOWN": "UNPROVEN", "NO CONCLUSION": "UNPROVEN",
    # 中文别名
    "支持": "TRUE", "属实": "TRUE",
    "反驳": "FALSE", "否定": "FALSE", "不成立": "FALSE", "驳斥": "FALSE",
    "证据不足": "UNPROVEN", "无法判定": "UNPROVEN", "无法确定": "UNPROVEN",
    "不确定": "UNPROVEN", "未证实": "UNPROVEN", "尚无定论": "UNPROVEN", "无法下结论": "UNPROVEN",
    "混合": "MIXTURE", "部分支持": "MIXTURE", "部分反驳": "MIXTURE", "部分属实": "MIXTURE",
}

def _normalize_label(s: str) -> str:
    if not s:
        return "ERROR"
    su = re.sub(r"\s+", " ", s.strip().upper())
    return ALIAS_MAP.get(su, su) if su in (set(ALIAS_MAP) | VALID_LABELS) else "ERROR"

# ===== 更稳健的解析器：优先匹配“最终结论”，再兜底扫描；最后再规则兜底 =====
def extract_final_label_from_output(output_text: str) -> str:
    if not isinstance(output_text, str) or not output_text.strip():
        return "ERROR"

    s = output_text.strip()
    su = s.upper()

    # 1) 明确的“最终结论”句式（英文/中文）
    concl_triggers_en = [
        r"(?:SO\s+)?THE\s+FINAL\s+ANSWER\s+IS\s*[:：]?\s*",
        r"(?:FINAL|CONCLUSION|LABEL|RESULT)\s*[:：]\s*",
        r"(?:FINAL\s+LABEL|FINAL\s+RESULT)\s*[:：]\s*",
    ]
    concl_trigger_zh = r"(?:最终|最后)?\s*(?:结论|答案|判断|标签|结果)\s*[:：]\s*"
    lbl_en = r"(TRUE|FALSE|UNPROVEN|MIXTURE|SUPPORTS?|REFUTES?|YES|NO|MIXED|PARTIAL SUPPORT|PARTLY TRUE|NOT ENOUGH EVIDENCE|INSUFFICIENT EVIDENCE|CANNOT BE DETERMINED|UNDETERMINED|UNKNOWN|NO CONCLUSION)"
    tail = r"(?:\s*(?:[\.\!\?，。；;：:]|$))"

    # 英文触发
    for pat in concl_triggers_en:
        mlist = list(re.finditer(pat + lbl_en + tail, su, flags=re.IGNORECASE))
        if mlist:
            tok = mlist[-1].group(1)
            lab = _normalize_label(tok)
            if lab in VALID_LABELS:
                return lab

    # 中文触发（匹配中文 alias）
    zh_keys = [k for k in ALIAS_MAP.keys() if re.search(r"[\u4e00-\u9fff]", k)]
    if zh_keys:
        zh_union = "|".join(map(re.escape, sorted(zh_keys, key=len, reverse=True)))
        mlist = list(re.finditer(concl_trigger_zh + f"({zh_union})" + tail, s, flags=re.IGNORECASE))
        if mlist:
            tok = mlist[-1].group(1)
            lab = _normalize_label(tok)
            if lab in VALID_LABELS:
                return lab

    # 2) 兜底：全文从后向前找“独立的标签/别名”
    mlist = list(re.finditer(lbl_en + tail, su, flags=re.IGNORECASE))
    if mlist:
        tok = mlist[-1].group(1)
        lab = _normalize_label(tok)
        if lab in VALID_LABELS:
            return lab

    if zh_keys:
        mlist = list(re.finditer(f"({zh_union})" + tail, s, flags=re.IGNORECASE))
        if mlist:
            tok = mlist[-1].group(1)
            lab = _normalize_label(tok)
            if lab in VALID_LABELS:
                return lab

    # 3) 超轻量规则兜底（不改提示词，仅解析层面）
    heur = su.replace("\n", " ")
    # 先判明显否定/反驳
    if re.search(r"\bREFUTE[SD]?\b|NO EVIDENCE SUPPORTING|DOES NOT SUPPORT|NOT TRUE|INCORRECT|FALSE\b", heur):
        return "FALSE"
    # 明显支持
    if re.search(r"\bSUPPORT[SED]?\b|CONSISTENT WITH|EVIDENCE SHOWS|CONFIRM(S|ED)?\b|TRUE\b", heur):
        return "TRUE"
    # 混合/部分
    if re.search(r"\bMIX(ED|TURE)\b|PARTIAL(LY)?\b|BOTH TRUE AND FALSE|SOME EVIDENCE\b", heur):
        return "MIXTURE"
    # 证据不足/不确定
    if re.search(r"INSUFFICIENT|NOT ENOUGH EVIDENCE|CANNOT BE DETERMINED|UNDETERMINED|UNKNOWN|NO CONCLUSION|INCONCLUSIVE", heur):
        return "UNPROVEN"

    return "ERROR"

# ===== 只重跑 Predicted_Label=ERROR 的行，不动其它 =====
INPUT_CSV  = r"C:\Users\sakur\Desktop\paper\Pubhealt4label-selfask\pubhealth_dev_combined_rerun_errors-2.csv"
OUTPUT_CSV = r"C:\Users\sakur\Desktop\paper\Pubhealt4label-selfask\pubhealth_dev_combined_rerun_errors-3.csv"

df = pd.read_csv(INPUT_CSV)
mask_err = df["Predicted_Label"].astype(str).str.upper().eq("ERROR")
todo = df[mask_err].copy()
print(f"待重跑（Predicted_Label=error）：{len(todo)} 条")

fixed = 0
for idx, row in todo.iterrows():
    claim = str(row["Claim"])
    evidence = str(row["Evidence"])

    # 用你的提示词构造同款 Self-Ask prompt
    prompt = build_self_ask_prompt(claim, evidence)

    # 兼容两种签名： (claim, evidence, prompt) 或 (claim, evidence)
    try:
        raw_output = gpt_self_ask_verifier(claim, evidence, prompt)  # 如果你的函数支持传 prompt
    except TypeError:
        raw_output = gpt_self_ask_verifier(claim, evidence)          # 否则退回原签名

    # 解析输出 → 覆盖 error
    new_label = extract_final_label_from_output(raw_output)
    if new_label in VALID_LABELS:
        df.at[idx, "Predicted_Label"] = new_label
        df.at[idx, "Raw_Output"] = raw_output
        fixed += 1
    else:
        # 至少把最新 raw_output 写回，方便你后续人工排查
        df.at[idx, "Raw_Output"] = raw_output

print(f"✅ 修复完成：成功覆盖 {fixed} 条；剩余 ERROR：{df['Predicted_Label'].astype(str).str.upper().eq('ERROR').sum()} 条")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"已保存到：{OUTPUT_CSV}")
