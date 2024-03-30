import streamlit.web.cli as stcli
import sys

# Shortcut to run streamlit app
def main():
    sys.argv = ["streamlit", "run", "streamlit_app.py"]
    sys.exit(stcli.main())

if __name__ == '__main__':
    main()
