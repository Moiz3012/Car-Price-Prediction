from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model parameters
w_final = np.load('model/w_final.npy')
b_final = np.load('model/b_final.npy')
X_mean = np.load('model/X_mean.npy')
X_std = np.load('model/X_std.npy')
columns_order = np.load('model/columns_order.npy', allow_pickle=True)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # 1️⃣ Get form data
    data = request.form.to_dict()

    # 2️⃣ Convert numeric fields properly
    data['Year'] = int(data['Year'])
    data['Present_Price'] = float(data['Present_Price'])
    data['Kms_Driven'] = int(data['Kms_Driven'])
    data['Owner'] = int(data['Owner'])

    # 3️⃣ Make dataframe
    df = pd.DataFrame([data])

    # 4️⃣ One-hot encode categorical columns like training
    df_encoded = pd.get_dummies(df, columns=['Car_Name','Fuel_Type','Seller_Type','Transmission'], drop_first=True)

    # 5️⃣ Align with original training columns
    columns_order = np.load('model/columns_order.npy', allow_pickle=True)
    df_encoded = df_encoded.reindex(columns=columns_order, fill_value=0)

    # 6️⃣ Load model params and scalers
    X_mean = np.load('model/X_mean.npy')
    X_std = np.load('model/X_std.npy')
    w_final = np.load('model/w_final.npy')
    b_final = np.load('model/b_final.npy')

    # 7️⃣ Normalize inputs
    X = (df_encoded.values - X_mean) / X_std

    # 8️⃣ Prediction
    prediction = np.dot(X, w_final) + b_final
    price = float(prediction[0])

    return render_template('index.html', prediction_text=f'Predicted Selling Price: {price:.2f} lakh')

if __name__ == '__main__':
    app.run(debug=True)
