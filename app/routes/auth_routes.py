from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid payload"}), 400

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    registration_pin = (data.get("pin") or "").strip()

    if not name or not email or not password or not registration_pin:
        return jsonify({"error": "Name, email, password, and registration PIN are required"}), 400

    if registration_pin != "121212":
        return jsonify({"error": "PIN pendaftaran tidak valid"}), 400

    db = current_app.config.get("MONGO_DB")
    if db is None:
        return jsonify({"error": "Database is not configured"}), 500

    users = db["users"]

    existing = users.find_one({"email": email})
    if existing:
        return jsonify({"error": "Email is already registered"}), 400

    password_hash = generate_password_hash(password)

    user_doc = {
        "name": name,
        "email": email,
        "password": password_hash,
        "created_at": datetime.utcnow(),
    }

    result = users.insert_one(user_doc)

    return (
        jsonify(
            {
                "message": "User registered successfully",
                "user": {
                    "id": str(result.inserted_id),
                    "name": name,
                    "email": email,
                },
            }
        ),
        201,
    )


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid payload"}), 400

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db = current_app.config.get("MONGO_DB")
    if db is None:
        return jsonify({"error": "Database is not configured"}), 500

    users = db["users"]
    user = users.find_one({"email": email})

    if not user or not check_password_hash(user.get("password", ""), password):
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify(
        {
            "message": "Login successful",
            "user": {
                "id": str(user["_id"]),
                "name": user.get("name"),
                "email": user.get("email"),
            },
        }
    )
