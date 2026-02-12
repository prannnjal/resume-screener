import os
import sys

DB_FILE = "recruiter.db"

def clean_database():
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
            print(f"✅ Database '{DB_FILE}' has been successfully deleted.")
            print("Restart the Streamlit app to initialize a fresh database.")
        except Exception as e:
            print(f"❌ Error deleting database: {e}")
    else:
        print(f"ℹ️ Database '{DB_FILE}' does not exist.")

if __name__ == "__main__":
    confirm = input(f"Are you sure you want to delete '{DB_FILE}'? This cannot be undone. (y/n): ")
    if confirm.lower() == 'y':
        clean_database()
    else:
        print("Operation cancelled.")
