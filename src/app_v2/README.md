# How to run CONNECT: 

## Steps
### 1. Clone the Repository
    git clone https://git.txstate.edu/DataLab/AI4Health.git
    cd AI4Health/CONNECT/SYSTEM

### 2. Setup the Env
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    using pip: pip install -r requirements.txt
    using Conda: conda create --name <env_name> --file requirements.txt

### 3. Running the app
    streamlit run app.py
    streamlit run modelinApp.py

