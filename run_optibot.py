import streamlit.web.cli as stcli
import sys

def main():
    # Modify the argv for Streamlit.
    sys.argv = ["streamlit", "run", "streamlit_app.py"]
    # Execute Streamlit's main function which starts the app.
    sys.exit(stcli.main())

if __name__ == '__main__':
    main()
