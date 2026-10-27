## Question
How much does the score move when only the wording of the instruction changes? Differences between experiments smaller than this spread are not real (task.md, "Prompt noise").

## Change vs baseline
Baseline: `c3_exp03_7b_v4` (Qwen2.5-7B-Instruct 4-bit, zero-shot, k = 0, definitions on, `template_v1.txt`; val macro-F1 0.207). Data v4. Notebook `notebooks/kaggle_c3_exp05-07.ipynb` (exp05, exp06 and exp07 in one session).
- Only change: the template, reworded twice with the same blanks, options and answer letters: `template_p1.txt` ("The following is an excerpt from a meeting transcript …"), `template_p2.txt` ("Read this extract from a recorded research meeting …").
- Run on the 7B at k = 0, the best setting so far (k = 1 is within 0.001).

## Runs
- `c3_exp07_7b_p1_v4`
- `c3_exp07_7b_p2_v4`

## Result
pending

## Conclusion
pending
