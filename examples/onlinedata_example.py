"""Example usage of OnlineData for fetching web data."""

from llm_validation_framework import OnlineData

if __name__ == "__main__":
    online_data = OnlineData()
    body, href = online_data.search("python programming")
    print(body, href)
