import pandas as pd
from agent import get_agent, run_query

def main():
    print("Creating sample dataframe...")
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [25, 30, 35, 40],
        "Salary": [50000, 60000, 70000, 80000]
    })
    
    print("Initializing agent...")
    agent, langfuse_handler = get_agent(df)
    
    print("Agent initialized successfully!")
    query = "What is the average salary?"
    print(f"Running query: '{query}'")
    
    try:
        response = run_query(agent, langfuse_handler, query)
        print("\n--- Response ---")
        print(response)
    except Exception as e:
        print(f"\nError running query: {e}")

if __name__ == "__main__":
    main()
