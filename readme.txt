FAKE REVIEW DETECTION SYSTEM (AI + HUMANIZED + URL MATCH)
==========================================================

PROJECT OVERVIEW
----------------
This project is an AI-based system for detecting Fake and Real reviews.
It uses a trained Machine Learning model (Logistic Regression) and NLP 
(TF-IDF Vectorization) to classify user-submitted reviews as "Fake" or "Real".

It also supports:
1. URL-based system using Selenium + BeautifulSoup to fetch live reviews.
2. Humanize system that compares AI and human perception match.
3. API endpoints for AI-based prediction and URL-based scraping.

----------------------------------------------------------
FEATURES
----------------------------------------------------------
1. AI-BASED REVIEW DETECTION
   - Logistic Regression model predicts "Fake" or "Real" for any review text.
   - Model returns probability score (Fake Percentage) for confidence.

2. HUMANIZE SYSTEM
   - Compares AI prediction with human opinion.
   - Displays AI vs Human match percentage for better transparency.

3. URL-BASED REVIEW MATCH SYSTEM (OPTIONAL)
   - Users can input a product or website URL.
   - The system uses Selenium and BeautifulSoup to fetch reviews.
   - All fetched reviews are analyzed by the AI model.
   - Output shows how many reviews are fake or real.
   - Example: Used for Amazon, Flipkart, or Hotel review pages.

4. MACHINE LEARNING MODEL
   - Model trained using Logistic Regression with TF-IDF Vectorization.
   - Dataset file: reviews.csv (contains text + label: "real" or "fake")
   - Model files: fake_model.pkl and fake_vectorizer.pkl
   - Accuracy improves with more and diverse data.

5. FLASK API INTEGRATION
   - REST API built using Flask.
   - Endpoints:
       /predict     : For direct review text prediction
       /predict_url : For URL-based review fetching + prediction
   - Handles JSON input/output.

6. DATA HANDLING
   - train.py script retrains model on updated dataset.
   - reviews.csv file can be extended with more labeled data.
   - After adding new data, run train.py again to update model.

7. ERROR HANDLING
   - Invalid input and missing URLs handled gracefully.
   - Detailed JSON error responses.
   - Logging and debugging support can be added.

8. FUTURE SCOPE
   - Add Hybrid Models (Logistic + Random Forest + XGBoost)
   - Add Sentiment Analysis features
   - Add Threshold tuning for precision vs recall
   - Integrate React or EJS frontend dashboard
   - Add automatic retraining pipeline

   MODEL TRAINING PROCESS
----------------------------------------------------------
1. Dataset: reviews.csv
   Columns: review, label
   Example:
       "This product is amazing", real
       "Worst item ever", fake

2. Steps:
   - Load data using pandas.
   - Split data into train/test.
   - Use TF-IDF Vectorizer to convert text to numeric vectors.
   - Train Logistic Regression model.
   - Save model and vectorizer as pickle files.

3. Retraining:
   - Add new reviews to reviews.csv.
   - Run: python train.py
   - This updates model.pkl and vectorizer.pkl with new data.


   1. POST /predict
   Description: Predicts whether given text review is Fake or Real.

   Input Example:
   {
     "review": "This phone is really good and battery lasts long."
   }

   Output Example:
   {
     "review": "This phone is really good and battery lasts long.",
     "AI": "Real",
     "Humanize": "Real"
   }

----------------------------------------------------------

2. POST /predict_url
   Description: Scrapes and analyzes reviews from product URL.

   Input Example:
   {
     "url": "https://www.amazon.in/dp/B0CXYZ12345"
   }

   Output Example:
   {
     "url": "https://www.amazon.in/dp/B0CXYZ12345",
     "reviews_checked": 8,
     "results": [
        {"review": "Excellent camera quality", "AI": "Real"},
        {"review": "Worst phone ever!", "AI": "Fake"}
     ]
   }

   Notes:
   - Selenium WebDriver must be installed and path configured.
   - Works for Amazon, Flipkart, and similar product review pages.
COMMANDS TO RUN
----------------------------------------------------------
1. Install all requirements:
   pip install -r requirements.txt

2. Train the model:
   python train.py

3. Run Flask API:
   python main.py

4. Test via Postman or CURL:
   curl -X POST -H "Content-Type: application/json" \
   -d '{"review": "Good product, worth the price"}' \
   http://127.0.0.1:5000/predict

   For URL based:
   curl -X POST -H "Content-Type: application/json" \
   -d '{"url": "https://www.amazon.in/dp/B0CXYZ12345"}' \
   http://127.0.0.1:5000/predict_url


TECHNOLOGIES USED
----------------------------------------------------------
Backend Framework  : Flask (Python)
Machine Learning    : Logistic Regression (Scikit-learn)
Vectorization       : TF-IDF
Scraping Tool       : Selenium + BeautifulSoup
Dataset Format      : CSV
Model Storage       : Pickle (.pkl)
Testing Tool        : Postman / CURL

AUTHOR DETAILS
----------------------------------------------------------
Developed By  : Lakshya Mittal
Role          : AI Developer | Full Stack Engineer
