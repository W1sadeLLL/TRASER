"""Synthetic regression cases; no private source examples or dictionaries."""

import unittest
from traser.data.privacy import rule_redact, mask_spans, normalize_placeholders, PUBLIC_FIELDS


class PrivacyContract(unittest.TestCase):
    def test_contact_and_profile_redaction_precedes_mentions(self):
        text = "mail person@example.invalid @ExampleUser https://weibo.com/u/000000 联系电话13800000000"
        result = rule_redact(text)
        self.assertIn("[CONTACT]", result)
        self.assertIn("[USER]", result)
        self.assertIn("[LINK]", result)
        for token in ("person@", "ExampleUser", "weibo.com", "13800000000"):
            self.assertNotIn(token, result)

    def test_detailed_address_and_overlapping_entities(self):
        self.assertIn("[ADDRESS]", rule_redact("住址：示例路123号2楼。"))
        self.assertEqual(mask_spans("abcdefg", [(1, 4, "USER"), (3, 6, "USER")]), "a[USER]g")

    def test_redaction_is_idempotent_and_retains_nonidentifying_text(self):
        text = "普通讨论。@ExampleUser: 请看https://example.invalid/post/1"
        result = rule_redact(text)
        self.assertEqual(result, rule_redact(result))
        self.assertIn("普通讨论。", result)
        self.assertEqual(normalize_placeholders("[[USER]] [ADDRESS]ACT]"), "[USER] [ADDRESS]")

    def test_only_declared_public_columns_remain(self):
        self.assertEqual(
            set(PUBLIC_FIELDS),
            {
                "source_row",
                "occurrence_id",
                "content",
                "publish_time",
                "publish_time_utc",
                "repost_count",
                "comment_count",
                "like_count",
            },
        )


if __name__ == "__main__":
    unittest.main()
