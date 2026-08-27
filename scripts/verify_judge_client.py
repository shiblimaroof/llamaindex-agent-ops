"""
scripts/verify_judge_client.py

Throwaway smoke test for evalops/judge/client.py -- confirms the Cerebras
call actually works end to end (real API call, real JSON parse) before
building schema.py/prompt.py on top of it. Not a pytest file.
"""


from dotenv import load_dotenv
load_dotenv()

from evalops.judge.client import call_judge

system_prompt = "You are a test responder. Reply with valid JSON only, no markdown fences."
user_prompt = 'Return exactly this JSON: {"status": "ok", "value": 42}'


print("Calling nvidia...")
result = call_judge(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    source_id="test",
    run_id="test-run-001",
)
print("Got response")

print(result)