# backend.py or prompt_utils.py (Add this block)

leave_policy_examples = """
Q: Can I take medical leave for a cold?
A:
{
  "decision": "Approved",
  "justification": "Clause 12 permits medical leave for minor illnesses including cold and flu.",
  "clause_references": [12]
}

Q: Can I take paid leave to take care of my child after surgery?
A:
{
  "decision": "Approved",
  "justification": "Clause 18 allows paid leave for taking care of immediate family during medical treatment.",
  "clause_references": [18]
}
"""

auto_insurance_examples = """
Q: Will my car damage be covered in a collision?
A:
{
  "decision": "Approved",
  "justification": "Clause 6 covers vehicle collision damages as long as the driver holds a valid license.",
  "clause_references": [6]
}

Q: Can I claim if I hit an animal?
A:
{
  "decision": "Rejected",
  "justification": "Clause 9 specifically excludes accidents involving animals from coverage.",
  "clause_references": [9]
}
"""

health_insurance_examples = """
Q: Will hospitalization for appendicitis be covered?
A:
{
  "decision": "Approved",
  "justification": "Clause 14 covers surgical hospitalization for acute conditions.",
  "clause_references": [14]
}

Q: Is dental treatment covered?
A:
{
  "decision": "Rejected",
  "justification": "Clause 22 excludes cosmetic and dental treatments.",
  "clause_references": [22]
}
"""

home_insurance_examples = """
Q: Will my insurance cover fire damage?
A:
{
  "decision": "Approved",
  "justification": "Clause 5 provides coverage for fire-related damages to insured property.",
  "clause_references": [5]
}

Q: Is theft covered in case of a break-in?
A:
{
  "decision": "Approved",
  "justification": "Clause 11 includes theft under covered perils if break-in evidence is documented.",
  "clause_references": [11]
}
"""

def get_few_shot_examples(policy_type):
    if policy_type == "Employee Leave Policy":
        return leave_policy_examples
    elif policy_type == "Auto Insurance":
        return auto_insurance_examples
    elif policy_type == "Health Insurance":
        return health_insurance_examples
    elif policy_type == "Home Insurance":
        return home_insurance_examples
    else:
        return ""
