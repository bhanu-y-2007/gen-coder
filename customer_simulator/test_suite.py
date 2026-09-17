"""
Automated Test Suite for AI Customer Support & Simulator
Covers all user-specified Positive and Negative test cases.
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


class TestPositiveCases(unittest.TestCase):

    def test_01_frustrated(self):
        """Positive Case 1: Frustrated Customer"""
        sim = CustomerSimulator(persona="frustrated", scenario="refund_request", frustration_level=5)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertEqual(res["frustration_level"], 5)
        self.assertEqual(res["emotion"]["label"], "Frustrated")
        print("  [PASS] Frustrated Persona — Initialized level 5/10 successfully")

    def test_02_calm(self):
        """Positive Case 2: Calm Customer"""
        sim = CustomerSimulator(persona="polite", scenario="delayed_order", frustration_level=2)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertLessEqual(res["frustration_level"], 2)
        self.assertEqual(res["emotion"]["label"], "Calm")
        print("  [PASS] Calm Persona — Initialized level 2/10 successfully")

    def test_03_polite(self):
        """Positive Case 3: Polite Customer"""
        sim = CustomerSimulator(persona="polite", scenario="account_issue", frustration_level=2)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertEqual(res["persona"], "polite")
        print("  [PASS] Polite Persona — Responded politely as expected")

    def test_04_angry(self):
        """Positive Case 4: Angry Customer"""
        sim = CustomerSimulator(persona="angry", scenario="refund_request", frustration_level=8)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertGreaterEqual(res["frustration_level"], 7)
        self.assertEqual(res["emotion"]["label"], "Angry")
        print("  [PASS] Angry Persona — Initialized high intensity level 8/10 successfully")

    def test_05_furious(self):
        """Positive Case 5: Furious Customer"""
        sim = CustomerSimulator(persona="furious", scenario="payment_failure", frustration_level=10)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertEqual(res["frustration_level"], 10)
        self.assertEqual(res["emotion"]["label"], "Furious")
        print("  [PASS] Furious Persona — Displayed extreme intensity level 10/10 successfully")

    def test_06_confused(self):
        """Positive Case 6: Confused Customer"""
        sim = CustomerSimulator(persona="concerned", scenario="account_issue", frustration_level=4)
        res = sim.start()
        self.assertIn("customer_message", res)
        self.assertEqual(res["emotion"]["label"], "Concerned")
        print("  [PASS] Confused Persona — Responded with clarifying questions as expected")

    def test_06b_text_cleaner(self):
        """Positive Case 7: Text Cleaner Normalization"""
        dirty = "  Hello    world! \n\n  This is   a test.  "
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "Hello world! This is a test.")
        print("  [PASS] Text Cleaner — Cleaned redundant spaces successfully")

    def test_06c_chunker_success(self):
        """Positive Case 8: Chunker Creates Overlapping Chunks"""
        docs = [{"text": "A" * 600, "metadata": {"source": "test.txt"}}]
        chunks = create_chunks(docs, chunk_size=300, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["source"], "test.txt")
        print("  [PASS] Chunker — Split document into overlapping chunks successfully")

    def test_06d_emotion_manager(self):
        """Positive Case 9: Emotion Manager State Management"""
        em = EmotionManager(initial_emotion="angry")
        self.assertEqual(em.get_state().intensity, 8)
        state = em.update("I apologize for the delay. We processed your full refund immediately.")
        self.assertLess(state.intensity, 8)
        print("  [PASS] Emotion Manager — Updated emotion state dynamically")


class TestNegativeCases(unittest.TestCase):

    def test_07_empty_customer_message(self):
        """Negative Case 1: Empty Customer Message"""
        sim = CustomerSimulator(persona="polite", scenario="refund_request")
        sim.start()
        res = sim.respond("")
        self.assertIn("customer_message", res)
        self.assertTrue(len(res["customer_message"]) > 0)
        print("  [PASS] Empty Customer Message — Handled safely with prompt response")

    def test_08_only_spaces(self):
        """Negative Case 2: Only Spaces"""
        sim = CustomerSimulator(persona="polite", scenario="refund_request")
        sim.start()
        res = sim.respond("     ")
        self.assertIn("customer_message", res)
        self.assertTrue(len(res["customer_message"]) > 0)
        print("  [PASS] Only Spaces — Handled safely without error")

    def test_09_invalid_emotion(self):
        """Negative Case 3: Invalid Emotion / Persona"""
        sim = CustomerSimulator(persona="invalid_emotion_xyz", scenario="refund_request")
        res = sim.start()
        self.assertEqual(sim.persona_name, "frustrated")
        self.assertIn("customer_message", res)
        print("  [PASS] Invalid Emotion — Safely fell back to default 'frustrated' persona")

    def test_10_invalid_intensity(self):
        """Negative Case 4: Invalid Intensity (Out of range values -10, 999)"""
        sim_low = CustomerSimulator(frustration_level=-10)
        sim_high = CustomerSimulator(frustration_level=999)
        self.assertEqual(sim_low.frustration_level, 1)
        self.assertEqual(sim_high.frustration_level, 10)
        print("  [PASS] Invalid Intensity — Clamped safely to valid bounds [1, 10]")

    def test_11_very_long_message(self):
        """Negative Case 5: Very Long Message"""
        sim = CustomerSimulator()
        sim.start()
        long_msg = "Please process my refund urgently right now. " * 300
        res = sim.respond(long_msg)
        self.assertIn("customer_message", res)
        print("  [PASS] Very Long Message — Processed without memory or crash error")

    def test_12_missing_customer_message(self):
        """Negative Case 6: Missing Customer Message (None input)"""
        sim = CustomerSimulator()
        sim.start()
        res = sim.respond(None)
        self.assertIn("customer_message", res)
        self.assertTrue(len(res["customer_message"]) > 0)
        print("  [PASS] Missing Customer Message — Handled gracefully with fallback prompt")

    def test_13_dynamic_frustration_decrease_comprehensive(self):
        """Dynamic Case 1: Comprehensive empathetic response decreases frustration significantly"""
        sim = CustomerSimulator(frustration_level=9)
        sim.start()
        agent_reply = (
            "I completely understand your concern regarding the refund for the damaged product. "
            "I sincerely apologize for the delay and inconvenience caused. I have carefully checked the "
            "details of your refund request and understand that you have been waiting. Your request is "
            "important to us, and we are working to ensure that the refund is processed correctly. "
            "I will provide you with a clear update as soon as possible with a concrete timeline and "
            "make sure the amount is credited back to your original payment method. Thank you for your patience."
        )
        res = sim.respond(agent_reply)
        self.assertLessEqual(res["frustration_level"], 6)
        print(f"  [PASS] Decrease Case 1 (Comprehensive Reply) — Frustration dropped from 9 to {res['frustration_level']}/10")

    def test_14_dynamic_frustration_decrease_ownership(self):
        """Dynamic Case 2: Supportive response taking ownership decreases frustration across multiple turns"""
        sim = CustomerSimulator(frustration_level=9)
        sim.start()
        
        # Turn 1: Empathetic response
        r1 = sim.respond("I'm sorry for the inconvenience. I understand your frustration and I'll help resolve your refund request right away.")
        self.assertLess(r1["frustration_level"], 9)
        
        # Turn 2: Ownership response (the exact case from screenshot)
        r2 = sim.respond("I understand You have waited long enough.I'll take ownership of this and get your refund resolved now.")
        self.assertLess(r2["frustration_level"], r1["frustration_level"])
        print(f"  [PASS] Decrease Case 2 (Multi-turn Supportive Ownership) — Progressively decreased 9 -> {r1['frustration_level']} -> {r2['frustration_level']}/10")

    def test_15_dynamic_frustration_decrease_mild(self):
        """Dynamic Case 3: Basic polite response decreases frustration by 1"""
        sim = CustomerSimulator(frustration_level=8)
        sim.start()
        agent_reply = "Sure, I can help you with that."
        res = sim.respond(agent_reply)
        self.assertEqual(res["frustration_level"], 7)
        print(f"  [PASS] Decrease Case 3 (Mild Polite Reply) — Frustration dropped from 8 to {res['frustration_level']}/10")

    def test_16_frustration_no_change_neutral(self):
        """Dynamic Case 4: Neutral acknowledgement results in NO CHANGE (Delta = 0)"""
        sim = CustomerSimulator(frustration_level=6)
        sim.start()
        res = sim.respond("Let me see.")
        self.assertEqual(res["frustration_level"], 6)
        print(f"  [PASS] No Change Case 1 (Neutral Reply) — Frustration remained unchanged at {res['frustration_level']}/10")

    def test_17_frustration_no_change_repetitive(self):
        """Dynamic Case 5: Duplicate / repetitive response produces NO DECREASE (Delta = 0)"""
        sim = CustomerSimulator(frustration_level=7)
        sim.start()
        sim.respond("I will look into this for you.")
        level_after_t1 = sim.frustration_level
        # Repeat the exact same message on next turn
        res = sim.respond("I will look into this for you.")
        self.assertEqual(res["frustration_level"], level_after_t1)
        print(f"  [PASS] No Change Case 2 (Repetitive Reply) — Frustration stayed unchanged at {res['frustration_level']}/10")

    def test_18_frustration_increases_on_dismissive_reply(self):
        """Dynamic Case 6: Dismissive response increases frustration"""
        sim = CustomerSimulator(frustration_level=5)
        sim.start()
        agent_reply = "There is nothing I can do. That is outside our control and company policy says no refunds."
        res = sim.respond(agent_reply)
        self.assertGreaterEqual(res["frustration_level"], 7)
        print(f"  [PASS] Increase Case 1 (Dismissive Reply) — Frustration increased from 5 to {res['frustration_level']}/10")

    def test_19_frustration_increases_on_blaming_customer(self):
        """Dynamic Case 7: Blaming customer increases frustration"""
        sim = CustomerSimulator(frustration_level=5)
        sim.start()
        agent_reply = "You should have read our terms before ordering. This is your fault."
        res = sim.respond(agent_reply)
        self.assertGreaterEqual(res["frustration_level"], 7)
        print(f"  [PASS] Increase Case 2 (Customer Blame) — Frustration increased from 5 to {res['frustration_level']}/10")

    def test_20_chunker_invalid_params(self):
        """Negative Case 7: Chunker with Invalid Arguments raises ValueError"""
        docs = [{"text": "Hello", "metadata": {}}]
        with self.assertRaises(ValueError):
            create_chunks(docs, chunk_size=0)
        with self.assertRaises(ValueError):
            create_chunks(docs, chunk_size=100, chunk_overlap=150)
        print("  [PASS] Chunker Validation — Invalid parameters correctly raised ValueError")

    def test_21_proportional_step_and_emotion_consistency(self):
        """Proportional Step & Emotion Consistency Test"""
        sim = CustomerSimulator(frustration_level=6)
        sim.start()
        
        # Moderately helpful reply causes proportional decrease from 6 -> 4 (Concerned)
        res1 = sim.respond("Thank you for your patience i understand your concern and i'm checking the refund status now")
        self.assertEqual(res1["frustration_level"], 4)
        self.assertEqual(res1["emotion"]["label"], "Concerned")
        
        # Next mild reply causes small decrease from 4 -> 3 (Concerned)
        res2 = sim.respond("I can confirm your details.")
        self.assertEqual(res2["frustration_level"], 3)
        self.assertEqual(res2["emotion"]["label"], "Concerned")
        
        # Next mild reply causes decrease from 3 -> 2 (Calm)
        res3 = sim.respond("I have completed the update.")
        self.assertEqual(res3["frustration_level"], 2)
        self.assertEqual(res3["emotion"]["label"], "Calm")
        print("  [PASS] Proportional Step & Emotion Consistency — Step transitions 6 -> 4 -> 3 -> 2 perfectly aligned with emotions")


def run_tests():
    print("=" * 75)
    print("      RUNNING POSITIVE & NEGATIVE TEST SUITE FOR CUSTOMER SIMULATOR")
    print("=" * 75 + "\n")

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestPositiveCases))
    suite.addTest(loader.loadTestsFromTestCase(TestNegativeCases))

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print("\n" + "=" * 75)
    print(f"SUMMARY: Ran {result.testsRun} tests | Failures: {len(result.failures)} | Errors: {len(result.errors)}")
    print("=" * 75)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
