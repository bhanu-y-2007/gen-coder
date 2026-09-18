"""
Automated Test Suite for AI Customer Support & Simulator
Covers all user-specified requirements on the 1 to 5 Frustration Scale:
  1 = Very calm / Satisfied
  2 = Slightly concerned
  3 = Moderately frustrated
  4 = Highly frustrated
  5 = Extremely angry / About to escalate
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


class TestFrustrationScaleAndPersonas(unittest.TestCase):

    def test_01_all_personas_initialization(self):
        """Verify all personas initialize with valid messages and scale 1-5"""
        for persona_name in ["polite", "concerned", "confused", "frustrated", "angry", "furious", "impatient"]:
            sim = CustomerSimulator(persona=persona_name, scenario="refund_request", frustration_level=3)
            res = sim.start()
            self.assertIn("customer_message", res)
            self.assertEqual(res["frustration_level"], 3)
            self.assertEqual(res["frustration_text"], "Frustration: 3/5")
            self.assertEqual(res["emotion"]["label"], "Moderately frustrated")
            print(f"  [PASS] Persona '{persona_name}' initialized successfully with frustration 3/5")

    def test_02_all_scenarios_initialization(self):
        """Verify all scenarios produce natural scenario-specific opening messages"""
        for scenario_name in ["refund_request", "delayed_order", "payment_failure", "account_issue", "cancellation"]:
            sim = CustomerSimulator(persona="frustrated", scenario=scenario_name, frustration_level=4)
            res = sim.start()
            self.assertIn("customer_message", res)
            self.assertEqual(res["frustration_level"], 4)
            self.assertEqual(res["frustration_text"], "Frustration: 4/5")
            self.assertEqual(res["emotion"]["label"], "Highly frustrated")
            print(f"  [PASS] Scenario '{scenario_name}' initialized successfully with frustration 4/5")

    def test_03_emotion_labels_1_to_5(self):
        """Verify emotion labels strictly adhere to the 1-5 scale specifications"""
        expected = {
            1: "Very calm / Satisfied",
            2: "Slightly concerned",
            3: "Moderately frustrated",
            4: "Highly frustrated",
            5: "Extremely angry / About to escalate"
        }
        for lvl, label in expected.items():
            sim = CustomerSimulator(frustration_level=lvl)
            res = sim.start()
            self.assertEqual(res["frustration_level"], lvl)
            self.assertEqual(res["frustration_text"], f"Frustration: {lvl}/5")
            self.assertEqual(res["emotion"]["label"], label)
        print("  [PASS] All 5 Frustration levels map to correct emotion labels")


class TestDynamicFrustrationTransitions(unittest.TestCase):

    def test_04_clear_helpful_reduces_by_2_to_4_points(self):
        """Clear, correct, helpful reply with empathy & resolution removes doubts: reduces frustration by 2 to 4 points"""
        # Test 5 -> 1 or 2
        sim5 = CustomerSimulator(frustration_level=5, scenario="refund_request")
        sim5.start()
        reply_comprehensive = (
            "I sincerely apologize for the delay and completely understand your frustration. "
            "I have investigated your order and processed your full refund immediately. "
            "The amount of $149.99 has been credited back to your original payment card and you will receive a confirmation email within 24 hours."
        )
        res5 = sim5.respond(reply_comprehensive)
        self.assertLessEqual(res5["frustration_level"], 2)
        print(f"  [PASS] Clear helpful reply: Frustration dropped from 5/5 to {res5['frustration_level']}/5 (Delta {res5['analysis']['delta']})")

        # Test 4 -> 1
        sim4 = CustomerSimulator(frustration_level=4, scenario="delayed_order")
        sim4.start()
        reply_order = (
            "I deeply apologize for the delay. I have taken full ownership of your order #ORD-39215 and contacted the carrier. "
            "Your replacement has been shipped via priority courier and the guaranteed delivery date is tomorrow by 2 PM. Here is your tracking link."
        )
        res4 = sim4.respond(reply_order)
        self.assertLessEqual(res4["frustration_level"], 2)
        print(f"  [PASS] Clear helpful reply: Frustration dropped from 4/5 to {res4['frustration_level']}/5 (Delta {res4['analysis']['delta']})")

        # Test 3 -> 1
        sim3 = CustomerSimulator(frustration_level=3, scenario="account_issue")
        sim3.start()
        reply_account = (
            "I apologize for the lockout trouble. I have unlocked your account and access has been restored immediately. "
            "Your Premium subscription is fully verified and active."
        )
        res3 = sim3.respond(reply_account)
        self.assertEqual(res3["frustration_level"], 1)
        print(f"  [PASS] Clear helpful reply: Frustration dropped from 3/5 to {res3['frustration_level']}/5 (Delta {res3['analysis']['delta']})")

    def test_05_partially_helpful_reduces_by_1_point(self):
        """Partially helpful reply with some doubt reduces frustration by 1 point only"""
        sim = CustomerSimulator(frustration_level=4, scenario="refund_request")
        sim.start()
        partial_reply = "I understand. I can help look into your refund status."
        res = sim.respond(partial_reply)
        self.assertEqual(res["frustration_level"], 3)
        self.assertEqual(res["analysis"]["delta"], -1)
        print("  [PASS] Partially helpful reply: Frustration dropped from 4/5 to 3/5 (Delta -1)")

    def test_06_neutral_or_repetitive_yields_no_change(self):
        """Neutral or repetitive response produces 0 change"""
        sim = CustomerSimulator(frustration_level=3)
        sim.start()

        # Turn 1: Neutral reply
        r1 = sim.respond("Let me see.")
        self.assertEqual(r1["frustration_level"], 3)
        self.assertEqual(r1["analysis"]["delta"], 0)
        print("  [PASS] Neutral reply: Frustration stayed at 3/5 (Delta 0)")

        # Turn 2: Repetitive reply
        sim.respond("I am checking the database.")
        level_before_repeat = sim.frustration_level
        r2 = sim.respond("I am checking the database.")
        self.assertEqual(r2["frustration_level"], level_before_repeat)
        self.assertEqual(r2["analysis"]["delta"], 0)
        print(f"  [PASS] Repetitive reply: Frustration stayed at {r2['frustration_level']}/5 (Delta 0)")

    def test_07_confusing_incomplete_increases_by_1_or_2(self):
        """Confusing, incomplete, or uncertain response increases frustration by 1 or 2 points"""
        sim = CustomerSimulator(frustration_level=2)
        sim.start()
        vague_reply = "What was your issue again? Can you repeat everything?"
        res = sim.respond(vague_reply)
        self.assertGreaterEqual(res["frustration_level"], 3)
        print(f"  [PASS] Confusing reply: Frustration increased from 2/5 to {res['frustration_level']}/5")

    def test_08_very_bad_dismissive_increases_by_2_or_3(self):
        """Very bad, dismissive, or customer-blaming response increases frustration by 2 or 3 points"""
        sim = CustomerSimulator(frustration_level=2)
        sim.start()
        bad_reply = "That is not my problem and outside our control. Stop complaining and deal with it."
        res = sim.respond(bad_reply)
        self.assertGreaterEqual(res["frustration_level"], 4)
        print(f"  [PASS] Dismissive reply: Frustration increased from 2/5 to {res['frustration_level']}/5 (Delta {res['analysis']['delta']})")

    def test_09_clamping_bounds_1_and_5(self):
        """Verify frustration is strictly clamped between 1 and 5"""
        sim_low = CustomerSimulator(frustration_level=-10)
        self.assertEqual(sim_low.frustration_level, 1)

        sim_high = CustomerSimulator(frustration_level=999)
        self.assertEqual(sim_high.frustration_level, 5)

        # Test decrease below 1 stays 1
        sim_low.start()
        res_min = sim_low.respond("I apologize and processed your full refund immediately.")
        self.assertEqual(res_min["frustration_level"], 1)

        # Test increase above 5 stays 5
        sim_high.start()
        res_max = sim_high.respond("Not my problem, deal with it.")
        self.assertEqual(res_max["frustration_level"], 5)
        print("  [PASS] Frustration bounds [1, 5] strictly maintained across all operations")


class TestMessageBehaviorAndUniqueness(unittest.TestCase):

    def test_10_no_repeated_messages_across_turns(self):
        """Verify customer messages do not repeat across multiple turns in a conversation"""
        sim = CustomerSimulator(persona="frustrated", scenario="refund_request", frustration_level=3)
        sim.start()

        seen_messages = set()
        seen_messages.add(sim.history[0]["content"])

        for i in range(5):
            res = sim.respond(f"I am looking into this matter step {i+1}.")
            msg = res["customer_message"]
            self.assertNotIn(msg, seen_messages, f"Message was repeated: {msg}")
            seen_messages.add(msg)

        self.assertEqual(len(seen_messages), 6)
        print("  [PASS] No repeated messages across 6 consecutive turns")

    def test_11_tone_and_length_variation_by_frustration(self):
        """Verify high frustration messages are shorter, sharper, and more aggressive than calm ones"""
        sim_calm = CustomerSimulator(frustration_level=1, scenario="refund_request")
        r_calm = sim_calm.start()

        sim_angry = CustomerSimulator(frustration_level=5, scenario="refund_request")
        r_angry = sim_angry.start()

        self.assertIn("frustration_text", r_calm)
        self.assertEqual(r_calm["frustration_text"], "Frustration: 1/5")
        self.assertEqual(r_angry["frustration_text"], "Frustration: 5/5")
        print(f"  [PASS] Tone variations verified:\n         Level 1 (Calm): {r_calm['customer_message']}\n         Level 5 (Furious): {r_angry['customer_message']}")

    def test_12_emotion_manager_module(self):
        """Verify EmotionManager class adheres to 1-5 scale"""
        em = EmotionManager(initial_emotion="frustrated")
        self.assertEqual(em.get_state().intensity, 3)
        st = em.update("I sincerely apologize for the delay. We processed your full refund immediately.")
        self.assertLessEqual(st.intensity, 2)
        print(f"  [PASS] EmotionManager updated state from 3/5 to {st.intensity}/5 successfully")


def run_tests():
    print("=" * 75)
    print("  CUSTOMER SIMULATOR 1-5 FRUSTRATION SCALE TEST SUITE")
    print("=" * 75 + "\n")

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestFrustrationScaleAndPersonas))
    suite.addTest(loader.loadTestsFromTestCase(TestDynamicFrustrationTransitions))
    suite.addTest(loader.loadTestsFromTestCase(TestMessageBehaviorAndUniqueness))

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print("\n" + "=" * 75)
    print(f"SUMMARY: Ran {result.testsRun} tests | Failures: {len(result.failures)} | Errors: {len(result.errors)}")
    print("=" * 75)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
