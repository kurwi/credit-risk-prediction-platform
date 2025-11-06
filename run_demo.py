import os
import sys

if __name__ == "__main__":
    # Launch Streamlit app for the credit risk demo
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    if not os.path.exists(app_path):
        print("Streamlit app not found at:", app_path)
        sys.exit(1)
    # Use the environment's streamlit executable
    os.system(f"streamlit run \"{app_path}\"")
