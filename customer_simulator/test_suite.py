"""
Automated Test Suite for AI Customer Support & Simulator
Runs Positive, Negative, Engine, and API test cases.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project paths to sys.path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR / "rag"))

from simulator import CustomerSimulator, PERSONAS, SCENARIOS, create_simulator
from emotion_manager import EmotionManager
from chunker import create_chunks
from text_cleaner import clean_text
from retriever import semantic_search


class TestCustomerSimulatorEngine(unittest.TestCase):

    def test_01_frustrated_customer(self):
        sim = CustomerSimulator(persona="frustrated", scenario="refund_request", frustration_level=5)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertEqual(res["frustration_level"], 5)
        self.assertFalse(res["finished"])
        print("  [PASS] Frustrated Customer initialized correctly")

    def test_02_calm_customer(self):
        sim = CustomerSimulator(persona="polite", scenario="delayed_order", frustration_level=2)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertLessEqual(res["frustration_level"], 3)
        print("  [PASS] Calm/Polite Customer initialized correctly")

    def test_03_polite_customer(self):
        sim = CustomerSimulator(persona="polite", scenario="account_issue", frustration_level=2)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertFalse(res["finished"])
        print("  [PASS] Polite Customer initialized correctly")

    def test_04_angry_customer(self):
        sim = CustomerSimulator(persona="angry", scenario="refund_request", frustration_level=8)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertGreaterEqual(res["frustration_level"], 7)
        print("  [PASS] Angry Customer initialized correctly")

    def test_05_furious_customer(self):
        sim = CustomerSimulator(persona="furious", scenario="payment_failure", frustration_level=10)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertEqual(res["frustration_level"], 10)
        print("  [PASS] Furious Customer initialized correctly")

    def test_06_confused_customer(self):
        sim = CustomerSimulator(persona="concerned", scenario="account_issue", frustration_level=4)
        res = sim.start()
        self.assertIn("customer_message", res)
        print("  [PASS] Confused/Concerned Customer initialized correctly")

    def test_07_deescalation_flow(self):
        sim = CustomerSimulator(persona="angry", scenario="refund_request", frustration_level=8)
        sim.start()
        res = sim.respond("I sincerely apologize for the delay. I have processed your full refund immediately.")
        self.assertLess(res["frustration_level"], 8)
        print("  [PASS] Deescalation flow lowers frustration intensity")

    def test_08_escalation_flow(self):
        sim = CustomerSimulator(persona="frustrated", scenario="refund_request", frustration_level=5)
        sim.start()
        res = sim.respond("Unfortunately policy says no refund. Please wait another 14 days.")
        self.assertGreaterEqual(res["frustration_level"], 5)
        print("  [PASS] Escalation flow retains/increases frustration on negative response")


class TestNegativeCases(unittest.TestCase):

    def test_09_empty_customer_message(self):
        sim = CustomerSimulator(persona="polite", scenario="refund_request")
        sim.start()
        res = sim.respond("")
        self.assertIn("customer_message", res)
        print("  [PASS] Empty message handled safely without crashing")

    def test_10_whitespace_message(self):
        sim = CustomerSimulator(persona="polite", scenario="refund_request")
        sim.start()
        res = sim.respond("     ")
        self.assertIn("customer_message", res)
        print("  [PASS] Whitespace-only message handled safely")

    def test_11_invalid_persona_fallback(self):
        sim = CustomerSimulator(persona="unknown_persona_123", scenario="refund_request")
        res = sim.start()
        self.assertEqual(sim.persona_name, "frustrated")
        print("  [PASS] Invalid persona safely fell back to default 'frustrated'")

    def test_12_invalid_frustration_bounds(self):
        sim_low = CustomerSimulator(frustration_level=-5)
        sim_high = CustomerSimulator(frustration_level=99)
        self.assertEqual(sim_low.frustration_level, 1)
        self.assertEqual(sim_high.frustration_level, 10)
        print("  [PASS] Out-of-bounds frustration levels bounded to [1, 10]")

    def test_13_very_long_message(self):
        sim = CustomerSimulator()
        sim.start()
        long_msg = "Please help me with my issue. " * 200
        res = sim.respond(long_msg)
        self.assertIn("customer_message", res)
        print("  [PASS] Very long input handled without crashing")


class TestRAGComponents(unittest.TestCase):

    def test_14_text_cleaner(self):
        raw = "  Hello   world! \n\n  This is   a test.  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Hello world! This is a test.")
        print("  [PASS] Text cleaner normalizes whitespace")

    def test_15_chunker_bounds(self):
        docs = [{"text": "Sample document content " * 50, "metadata": {"source": "test.txt"}}]
        chunks = create_chunks(docs, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        print("  [PASS] Document chunker creates valid chunks")

    def test_16_semantic_search(self):
        results = semantic_search("refund policy", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("text", results[0])
        print(f"  [PASS] Semantic search retrieved {len(results)} relevant results")


def run_tests():
    print("=" * 70)
    print("      RUNNING AUTOMATED TEST SUITE FOR CUSTOMER SIMULATOR & RAG")
    print("=" * 70 + "\n")

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestCustomerSimulatorEngine))
    suite.addTest(loader.loadTestsFromTestCase(TestNegativeCases))
    suite.addTest(loader.loadTestsFromTestCase(TestRAGComponents))

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"SUMMARY: Ran {result.testsRun} tests | Failures: {len(result.failures)} | Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
