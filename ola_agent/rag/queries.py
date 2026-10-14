"""Fixed query sets used by calibration, the grounded-answer demo and retrieval evaluation.

RELEVANT_DOCS was labelled by hand before any retrieval run, from the KB text.
"""

IN_SCOPE = [
    "How quickly will I get a reply to a P1 ticket?",
    "I was charged twice for one ride, can I get a refund?",
    "When does a ticket get escalated to Tier 2?",
    "How long do you keep my support chat history?",
    "What compensation do I get if my driver is very late?",
]

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Can you recommend a good biryani recipe?",
    "How do I reset my Netflix password?",
]

# The one out-of-scope query the Task 4 demo must answer with the fallback.
DEMO_OUT_OF_SCOPE = OUT_OF_SCOPE[0]

RELEVANT_DOCS = {
    IN_SCOPE[0]: {"KB-02"},
    IN_SCOPE[1]: {"KB-04"},
    IN_SCOPE[2]: {"KB-03"},
    IN_SCOPE[3]: {"KB-12"},
    IN_SCOPE[4]: {"KB-08", "KB-04"},
}
