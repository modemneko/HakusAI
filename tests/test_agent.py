import unittest
from src.main import process_message

class TestAgent(unittest.TestCase):
    def test_basic_response(self):
        response = process_message("test_user", "你好")
        self.assertTrue(isinstance(response, str))
        self.assertNotEqual(response, "")

    def test_search_tool(self):
        response = process_message("test_user", "今天北京天气")
        self.assertIn("天气", response)

if __name__ == "__main__":
    unittest.main()