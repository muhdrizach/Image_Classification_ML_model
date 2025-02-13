import joblib
import pickle
import requests
from io import BytesIO
from skimage.io import imread
from skimage.transform import resize
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
#model = pickle.load(open('img_model.p', 'rb'))  # Ensure this is the correct path to your model
# with open("img_model.p", "rb") as f:
#     model = pickle.load(f)

model = joblib.load('img_model.joblib')


# Categories (replace with actual categories from your model)
Categories = ['Aeroplanes','Cars']

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_result = None
    image_url = None
    if request.method == "POST":
        # Get the image URL from the form
        image_url = request.form["image_url"]
        
        try:
            # Fetch image from URL
            response = requests.get(image_url)
            
            if response.status_code == 200:
                # Load the image using imread
                img_data = imread(BytesIO(response.content))

                # Preprocess the image (resize and flatten)
                img_resized = resize(img_data, (150, 150, 3))  # Adjust size as needed
                img_flattened = img_resized.flatten().reshape(1, -1)  # Flatten the image

                # Predict probabilities
                probability = model.predict_proba(img_flattened)

                # Prepare the result
                prediction_result = {}
                for ind, category in enumerate(Categories):
                    prediction_result[category] = f'{probability[0][ind] * 100:.2f}%'

                # Predicted category
                predicted_category = Categories[model.predict(img_flattened)[0]]
                prediction_result["predicted_category"] = predicted_category
            else:
                prediction_result = {"error": "Failed to fetch the image. Please check the URL."}

        except Exception as e:
            prediction_result = {"error": f"Error processing image: {str(e)}"}

    return render_template("index.html", prediction_result=prediction_result, image_url=image_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
