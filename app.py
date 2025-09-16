from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from React Native frontend

# Dummy users for testing
users = [
    {"email": "test@example.com", "password": "123456", "department": "water"},
    {"phone": "9876543210", "password": "123456", "department": "electricity"},
]

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")
    password = data.get("password")

    if not password:
        return jsonify({"success": False, "message": "Password is required"}), 400

    # Email login
    if email:
        user = next((u for u in users if u.get("email") == email and u["password"] == password), None)
        if user:
            return jsonify({"success": True, "message": "Login successful"})

    # Phone login
    if phone:
        user = next((u for u in users if u.get("phone") == phone and u["password"] == password), None)
        if user:
            return jsonify({"success": True, "message": "Login successful"})

    return jsonify({"success": False, "message": "Invalid credentials"}), 401


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
