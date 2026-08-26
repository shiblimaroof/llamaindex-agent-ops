line = "@@ -12,3 +20,5 @@ def foo():"

print("Original line:")
print(line)

step1 = line.split("+", 1)
print("\n1. split('+', 1):")
print(step1)

step2 = step1[1]
print("\n2. Take [1]:")
print(step2)

step3 = step2.split("@@")
print("\n3. split('@@'):")
print(step3)

step4 = step3[0]
print("\n4. Take [0]:")
print(step4)

step5 = step4.strip()
print("\n5. strip():")
print(step5)

new_part = step5

print("\nFinal new_part:")
print(new_part)


import re

text = 'File "llama_index/core/foo.py", line 42'

match = re.search(r'File "([^"]+\.py)"', text)

print(match.group(0))
print(match.group(1))