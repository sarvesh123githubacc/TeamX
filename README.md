# 📰 AI-Based Fake News Analyzing Browser Extension

## 📌 Overview
Fake news is a significant issue in today's digital world. Our browser extension leverages AI to analyze and designed to detect and analyze fake news articles in real-time. The extension integrates Natural Language Processing (NLP) and Machine Learning (ML) to assess the credibility of news articles and provide users with a trustworthiness score.
 
🎯 Key Features
## ✨ Features
✅ **Real-time Fake News Detection** – Analyzes web content and assigns a credibility score.  
✅ **Machine Learning Model** – Utilizes a trained Bidirectional LSTM model for classification.  
✅ **Browser Integration** – Works as a Chrome/Firefox extension for seamless browsing.  
✅ **Interactive UI** – Displays warnings and trust scores for news articles.

## 🛠️ Tech Stack
- **Frontend**: Vite + React
- **Backend**: Flask (Python API)
- **Machine Learning**: TensorFlow, Keras, Bidirectional LSTM
- **Data Handling**: Pandas, NumPy, NLTK, Scikit-learn

## 🚀 Installation & Setup

🔧 Prerequisites

Google Chrome or Firefox

Python 3.8+

Pip & Virtual Environment

🛠️ Steps to Install

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/sarvesh123githubacc/TeamX
cd fake-news-extension
```

### **2️⃣ Install Backend Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Train & Save the Model**
Run the following script to train the fake news classifier:
```bash
python predictions.ipynb
```
This will generate `predictions.pkl` in the project folder.

### **4️⃣ Run the Flask API**
```bash
python app.py
```
The API will be available at `http://localhost:5000/predict`.

### **5️⃣ Load the Extension in Chrome**
1. Open **Chrome** and go to `chrome://extensions/`
2. Enable **Developer Mode** (top-right corner)
3. Click **Load Unpacked** and select the `extension` folder.
4. The extension is now ready to use!

## 📌 How It Works
🎥 Demo Video
📌 [https://drive.google.com/file/d/1tpLe3GoQFMWZLBBP5Pezq8f8ZHZ48iWX/view]

## 🤝 Contributing
Want to improve this project? Follow these steps:
1. **Fork the repository**
2. **Create a new branch**
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Added a new feature"
   ```
4. **Push to your branch**
   ```bash
   git push origin feature-branch
   ```
5. **Open a Pull Request** 🚀

## 🔗 Future Enhancements
- ✅ Deploy the Flask API online (e.g., **Render, AWS, Hugging Face Spaces**)
- ✅ Improve model accuracy 
- ✅ Add **Fact-checking API integration** (Snopes, Google Fact Check)
- ✅ Build a **Firefox version** of the extension

## 📜 License
📄 This project is licensed under the **MIT License**.

---
🚀 **Developed by [TeamX] for FOSS Hack 2025** 🚀

