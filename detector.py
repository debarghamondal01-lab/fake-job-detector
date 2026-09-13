import pickle
import os

# Load the saved model and vectorizer
model = pickle.load(open('fake_job_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def predict_job(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    if prediction == 1:
        return "FAKE", probability[1] * 100
    else:
        return "REAL", probability[0] * 100

def main():
    print("=" * 50)
    print("   FAKE JOB POSTING DETECTOR")
    print("=" * 50)
    print("Paste a job description below.")
    print("When you are finished, type 'END' on a new line and press Enter.")
    print("Type 'quit' to exit.\n")
    
    while True:
        print("-" * 50)
        print("Paste job description (type END when done):")
        
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            if line.strip().upper() == 'QUIT':
                print("Goodbye!")
                return
            lines.append(line)
        
        text = " ".join(lines)
        
        if len(text.strip()) < 20:
            print("Please paste more text.")
            continue
        
        result, confidence = predict_job(text)
        print(f"\n>>> Result: {result}")
        print(f">>> Confidence: {confidence:.2f}%\n")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
