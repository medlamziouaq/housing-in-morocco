# Housing in Morocco 🏡🇲🇦  
**Real Estate Price Prediction using Machine Learning**  

## 📌 Overview  
This project aims to predict real estate prices in Morocco using machine learning techniques. By leveraging data on property characteristics such as location, type, size, and number of rooms, we trained a predictive model to estimate housing prices.  

## 📊 Data Science Workflow  
1. **Data Collection & Preprocessing**  
   - The dataset includes real estate listings from various Moroccan cities (Casablanca, Marrakech, Rabat, Agadir, Fez, Tangier).  
   - Preprocessing includes handling missing values, feature engineering, and encoding categorical variables.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualizing price distributions, correlations, and trends using Python libraries like `pandas`, `matplotlib`, and `seaborn`.  

3. **Machine Learning Model**  
   - Implemented a regression model to predict property prices.  
   - Used feature selection and hyperparameter tuning for optimization.  
   - Saved the trained model using `joblib` for deployment.  

4. **Deployment with Flask**  
   - Built a web interface with `Flask` to allow users to input property details and get real-time price predictions.  
   - Implemented dropdown menus for city and property type selection to avoid user input errors.
  ## 🛠 Technologies Used

**Programming Languages**  
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
<img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">

**Core Libraries**  
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"> 
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"> 
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">

**Web Technologies**  
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">

**Model Serialization**  
<img src="https://img.shields.io/badge/joblib-007396?style=for-the-badge" alt="Joblib">

## 🚀 How to Run the Project  
Run the following commands in your terminal (Linux/macOS) or PowerShell (Windows):  

```bash
# 1. Clone the repository and navigate to the project folder
git clone https://github.com/medlamziouaq/housing-in-morocco.git
cd housing-in-morocco

# 2. Install dependencies (create a virtual environment if needed)
pip install -r requirements.txt

# 3. Launch the Flask app
python app.py

# 4. Access the web app (CTRL + Click the link)
echo "🌐 Open http://127.0.0.1:5000/ in your browser"
