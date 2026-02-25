"""Example usage of OnlineData for fetching web data."""

from OnlineData import OnlineData

if __name__ == "__main__":
    online_data = OnlineData()
    body, href = online_data.search("python programming")
    print(body, href)