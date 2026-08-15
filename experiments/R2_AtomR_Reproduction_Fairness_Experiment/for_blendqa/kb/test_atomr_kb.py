import requests
import json
import time

PARSER_URL = "https://viskop.xlore.cn/programApi"
ENGINE_URL = "https://viskop.xlore.cn/large"

question = "What is the capital of France?"

print("=" * 60)
print("STEP 1: Test KoPL semantic parser")
print("URL:", PARSER_URL)
print("Question:", question)

try:
    start = time.time()

    response = requests.post(
        PARSER_URL,
        data={"question": question},
        timeout=30
    )

    print("Status:", response.status_code)
    print("Time:", round(time.time() - start, 2), "s")
    print("Response:")
    print(response.text[:3000])

    response.raise_for_status()

    data = response.json()
    program = data["program"]

except Exception as e:
    print("\n[PARSER FAILED]")
    print(type(e).__name__, str(e))
    raise SystemExit(1)


print("\n" + "=" * 60)
print("STEP 2: Test KoPL KB engine")
print("URL:", ENGINE_URL)
print("Program:")
print(json.dumps(program, indent=2, ensure_ascii=False))

try:
    start = time.time()

    response = requests.post(
        ENGINE_URL,
        json={"program": program},
        timeout=60
    )

    print("Status:", response.status_code)
    print("Time:", round(time.time() - start, 2), "s")
    print("Response:")
    print(response.text[:5000])

    response.raise_for_status()

except Exception as e:
    print("\n[ENGINE FAILED]")
    print(type(e).__name__, str(e))
    raise SystemExit(1)

print("\n" + "=" * 60)
print("Both AtomR KB services are reachable.")