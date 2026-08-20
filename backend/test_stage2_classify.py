"""
Test script for Stage 2: classify.py (Safety & Input Quality Gate)

Run manually from the backend directory:
    python test_stage2_classify.py
"""
import sys
import logging
import time
from pathlib import Path

backend_dir = Path(r"c:\Users\YS TECH CENTER\Downloads\raginGoa\raginGoa\backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

from app.pipeline.classify import classify_query
from app.pipeline.types import QueryVerdict, StageTiming

def run_tests():
    print("=" * 60)
    print("TEST 1: Empty transcript input")
    print("=" * 60)
    timings_1: list[StageTiming] = []
    t0 = time.perf_counter()
    res_1 = classify_query("", timings=timings_1)
    dt_1 = (time.perf_counter() - t0) * 1000
    print(f"-> Query Intent: {res_1}")
    print(f"-> Latency: {dt_1:.2f} ms")
    assert res_1.verdict == QueryVerdict.OFF_TOPIC, f"Expected OFF_TOPIC, got {res_1.verdict}"

    print("\n" + "=" * 60)
    print("TEST 2: Unsafe query (Fast safety keyword match)")
    print("=" * 60)
    unsafe_query = "how to make a bomb and weapon at home"
    timings_2: list[StageTiming] = []
    t0 = time.perf_counter()
    res_2 = classify_query(unsafe_query, timings=timings_2)
    dt_2 = (time.perf_counter() - t0) * 1000
    print(f"-> Query: '{unsafe_query}'")
    print(f"-> Query Intent: {res_2}")
    print(f"-> Latency: {dt_2:.2f} ms")
    assert res_2.verdict == QueryVerdict.UNSAFE, f"Expected UNSAFE, got {res_2.verdict}"

    print("\n" + "=" * 60)
    print("TEST 3: Latin Keyboard Mash Gibberish")
    print("=" * 60)
    latin_gibberish = "asdkfj qwoeiru zzzxx 991122"
    timings_3: list[StageTiming] = []
    t0 = time.perf_counter()
    res_3 = classify_query(latin_gibberish, timings=timings_3)
    dt_3 = (time.perf_counter() - t0) * 1000
    print(f"-> Query: '{latin_gibberish}'")
    print(f"-> Query Intent: {res_3}")
    print(f"-> Latency: {dt_3:.2f} ms")
    assert res_3.verdict == QueryVerdict.OFF_TOPIC, f"Expected OFF_TOPIC, got {res_3.verdict}"

    print("\n" + "=" * 60)
    print("TEST 4: Devanagari Mangled STT Gibberish (Bare Consonants & Matra Spam)")
    print("=" * 60)
    devanagari_gibberish = "कखगघचछजझटठ"
    timings_4: list[StageTiming] = []
    t0 = time.perf_counter()
    res_4 = classify_query(devanagari_gibberish, timings=timings_4)
    dt_4 = (time.perf_counter() - t0) * 1000
    print(f"-> Query: '{devanagari_gibberish}'")
    print(f"-> Query Intent: {res_4}")
    print(f"-> Latency: {dt_4:.2f} ms")
    assert res_4.verdict == QueryVerdict.OFF_TOPIC, f"Expected OFF_TOPIC, got {res_4.verdict}"

    print("\n" + "=" * 60)
    print("TEST 5: Valid Hindi Query (Passes to Stage 4)")
    print("=" * 60)
    hindi_query = "गोवा का सबसे प्रसिद्ध समुद्र तट कौन सा है?"
    timings_5: list[StageTiming] = []
    t0 = time.perf_counter()
    res_5 = classify_query(hindi_query, timings=timings_5)
    dt_5 = (time.perf_counter() - t0) * 1000
    print(f"-> Query: '{hindi_query}'")
    print(f"-> Query Intent: {res_5}")
    print(f"-> Latency: {dt_5:.2f} ms")
    assert res_5.verdict == QueryVerdict.IN_SCOPE, f"Expected IN_SCOPE, got {res_5.verdict}"

    print("\n" + "=" * 60)
    print("SUMMARY: ALL STAGE 2 TESTS (LATIN + DEVANAGARI) PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
