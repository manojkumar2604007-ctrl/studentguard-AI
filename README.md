# StudentGuard AI - Student Dropout Prediction using ANN

## Setup & run (VS Code terminal)
    python -m venv venv
    venv\Scripts\activate            (Windows)   |   source venv/bin/activate   (Linux/Mac)
    pip install -r requirements.txt
    python -m ml.train               # trains the ANN, writes models/ + metrics.json
    python test_prediction.py        # console prediction on a real held-out student
    python app.py                    # open http://127.0.0.1:5000

Dataset: UCI "Predict Students' Dropout and Academic Success" -> dataset/data.csv (semicolon separated, column `Target`).