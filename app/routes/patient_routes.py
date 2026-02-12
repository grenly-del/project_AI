from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from bson import ObjectId
from urllib.parse import unquote
import pytz


patient_bp = Blueprint("patient", __name__, url_prefix="/api")


def get_db():
    db = current_app.config.get("MONGO_DB")
    if db is None:
        raise RuntimeError("MongoDB is not configured")
    return db


def get_wita_time():
    """Get current time in WITA (Central Indonesia Time)"""
    wita = pytz.timezone('Asia/Makassar')
    return datetime.now(wita)

def generate_patient_id(db):
    """Generate patient ID in format p-001, p-002, etc."""
    patients_col = db["patients"]
    # Get all patients with patientId
    all_patients = list(patients_col.find(
        {"patientId": {"$exists": True, "$ne": None}},
        {"patientId": 1}
    ))
    
    max_num = 0
    for patient in all_patients:
        patient_id = patient.get("patientId", "")
        if patient_id.startswith("p-"):
            try:
                num = int(patient_id.split("-")[-1])
                if num > max_num:
                    max_num = num
            except (ValueError, IndexError):
                continue
    
    new_num = max_num + 1
    return f"p-{new_num:03d}"

def serialize_patient(doc):
    return {
        "id": str(doc.get("_id")),
        "patientId": doc.get("patientId"),  # Add patientId field
        "name": doc.get("name"),
        "age": doc.get("age"),
        "gender": doc.get("gender"),
        "phone": doc.get("phone"),
        "address": doc.get("address"),
        "notes": doc.get("notes"),
        "createdAt": doc.get("createdAt").isoformat() if doc.get("createdAt") else None,
        "totalScans": doc.get("totalScans", 0),
        "lastScanDate": doc.get("lastScanDate").isoformat() if doc.get("lastScanDate") else None,
    }


def serialize_scan(doc):
    return {
        "id": str(doc.get("_id")),
        "patientId": doc.get("patientId"),
        "prediction": doc.get("prediction"),
        "confidence": doc.get("confidence"),
        "imagePath": doc.get("imagePath"),
        "gradCamPath": doc.get("gradCamPath"),
        "annotatedImagePath": doc.get("annotatedImagePath"),
        "modelVersion": doc.get("modelVersion"),
        "scanDate": doc.get("scanDate").isoformat() if doc.get("scanDate") else None,
        "notes": doc.get("notes"),
        "pdfReportPath": doc.get("pdfReportPath"),
    }


@patient_bp.route("/patients", methods=["GET", "POST"])
def patients_collection():
    db = get_db()
    patients_col = db["patients"]

    if request.method == "GET":
        search = (request.args.get("q") or "").strip().lower()

        query = {}
        if search:
            # Simple case-insensitive search on name or stringified _id
            query = {
                "$or": [
                    {"name": {"$regex": search, "$options": "i"}},
                    {"_id": {"$regex": search, "$options": "i"}},
                ]
            }

        # Note: ObjectId is not directly regex-searchable; this is a simple placeholder.
        # For correctness, we only apply regex on name here.
        if search:
            query = {"name": {"$regex": search, "$options": "i"}}

        patients = [serialize_patient(p) for p in patients_col.find(query).sort("createdAt", -1)]
        return jsonify(patients)

    # POST - create new patient
    data = request.get_json() or {}

    name = (data.get("name") or "").strip()
    age = data.get("age")
    gender = (data.get("gender") or "").strip()
    phone = (data.get("phone") or "").strip()
    address = (data.get("address") or "").strip()
    notes = (data.get("notes") or "").strip()

    if not name or age is None or not gender:
        return jsonify({"error": "Name, age, and gender are required"}), 400

    try:
        age = int(age)
    except (TypeError, ValueError):
        return jsonify({"error": "Age must be a number"}), 400

    # Generate patient ID
    patient_id = generate_patient_id(db)
    
    # Use WITA time
    now = get_wita_time()
    
    patient_doc = {
        "patientId": patient_id,  # Add patientId field
        "name": name,
        "age": age,
        "gender": gender,
        "phone": phone,
        "address": address,
        "notes": notes,
        "createdAt": now,
        "totalScans": 0,
        "lastScanDate": None,
    }

    result = patients_col.insert_one(patient_doc)
    patient_doc["_id"] = result.inserted_id

    return jsonify(serialize_patient(patient_doc)), 201


@patient_bp.route("/patients/<patient_id>", methods=["GET"])
def get_patient_detail(patient_id: str):
    db = get_db()
    patients_col = db["patients"]
    scans_col = db["scans"]

    # URL decode patient_id in case it was encoded
    patient_id = unquote(patient_id)
    current_app.logger.info(f"Looking for patient with ID: {patient_id}")

    # Try to find patient by ObjectId first
    patient = None
    try:
        obj_id = ObjectId(patient_id)
        patient = patients_col.find_one({"_id": obj_id})
        if patient:
            current_app.logger.info(f"Patient found by ObjectId: {patient_id}")
    except Exception as e:
        current_app.logger.debug(f"Not a valid ObjectId: {e}")
        # If not a valid ObjectId, try to find by patientId (p-001, etc.)
        patient = patients_col.find_one({"patientId": patient_id})
        if patient:
            current_app.logger.info(f"Patient found by patientId: {patient_id}")
    
    # If still not found, try as string _id
    if not patient:
        try:
            patient = patients_col.find_one({"_id": patient_id})
            if patient:
                current_app.logger.info(f"Patient found by string _id: {patient_id}")
        except Exception as e:
            current_app.logger.debug(f"Error trying string _id: {e}")
    
    if not patient:
        current_app.logger.warning(f"Patient not found: {patient_id}")
        # Log all available patient IDs for debugging
        all_patients = list(patients_col.find({}, {"_id": 1, "patientId": 1}).limit(10))
        current_app.logger.debug(f"Sample patient IDs in DB: {[(str(p.get('_id')), p.get('patientId')) for p in all_patients]}")
        return jsonify({"error": "Patient not found"}), 404

    # Get patient's MongoDB _id for scan lookup
    patient_mongo_id = str(patient.get("_id"))
    
    # Find scans by patientId (MongoDB _id as string)
    scans = [
        serialize_scan(s)
        for s in scans_col.find({"patientId": patient_mongo_id}).sort("scanDate", -1)
    ]

    return jsonify({"patient": serialize_patient(patient), "scans": scans})


@patient_bp.route("/scans", methods=["POST"])
def create_scan():
    db = get_db()
    scans_col = db["scans"]
    patients_col = db["patients"]

    data = request.get_json() or {}

    patient_id = (data.get("patientId") or "").strip()
    prediction = (data.get("prediction") or "").strip()
    confidence = data.get("confidence")
    image_path = (data.get("imagePath") or "").strip()
    gradcam_path = (data.get("gradCamPath") or "").strip() or None
    annotated_path = (data.get("annotatedImagePath") or "").strip()
    model_version = (data.get("modelVersion") or "").strip()
    notes = (data.get("notes") or "").strip()
    pdf_report_path = (data.get("pdfReportPath") or "").strip() or None

    if not patient_id or not prediction or confidence is None or not image_path:
        return jsonify({"error": "patientId, prediction, confidence, and imagePath are required"}), 400

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return jsonify({"error": "confidence must be a number"}), 400

    # scanDate: gunakan yang dikirim frontend jika ada, jika tidak pakai sekarang (WITA)
    scan_date_raw = data.get("scanDate")
    if scan_date_raw:
        try:
            scan_date = datetime.fromisoformat(scan_date_raw)
            # Convert to WITA if not already timezone-aware
            if scan_date.tzinfo is None:
                wita = pytz.timezone('Asia/Makassar')
                scan_date = wita.localize(scan_date)
        except Exception:
            scan_date = get_wita_time()
    else:
        scan_date = get_wita_time()

    scan_doc = {
        "patientId": patient_id,
        "prediction": prediction,
        "confidence": confidence,
        "imagePath": image_path,
        "gradCamPath": gradcam_path,
        "annotatedImagePath": annotated_path,
        "modelVersion": model_version,
        "scanDate": scan_date,
        "notes": notes,
        "pdfReportPath": pdf_report_path,
    }

    result = scans_col.insert_one(scan_doc)
    scan_doc["_id"] = result.inserted_id

    # Update patient aggregate info
    # patient_id could be MongoDB _id (ObjectId string) or patientId (p-001)
    try:
        # Try to find patient first
        patient_obj = None
        try:
            obj_id = ObjectId(patient_id)
            patient_obj = patients_col.find_one({"_id": obj_id})
        except Exception:
            # If not ObjectId, try to find by patientId
            patient_obj = patients_col.find_one({"patientId": patient_id})
        
        if patient_obj:
            patient_mongo_id = patient_obj.get("_id")
            patients_col.update_one(
                {"_id": patient_mongo_id},
                {"$inc": {"totalScans": 1}, "$set": {"lastScanDate": scan_date}},
            )
    except Exception as e:
        # Log error but don't fail the scan creation
        current_app.logger.warning(f"Failed to update patient aggregate: {e}")
        pass

    return jsonify(serialize_scan(scan_doc)), 201


@patient_bp.route("/scans/<scan_id>", methods=["GET"])
def get_scan_detail(scan_id: str):
    db = get_db()
    scans_col = db["scans"]

    # URL decode scan_id in case it was encoded
    scan_id = unquote(scan_id)
    current_app.logger.info(f"Looking for scan with ID: {scan_id}")

    # Try to find scan by ObjectId
    scan = None
    try:
        obj_id = ObjectId(scan_id)
        scan = scans_col.find_one({"_id": obj_id})
        if scan:
            current_app.logger.info(f"Scan found by ObjectId: {scan_id}")
    except Exception as e:
        current_app.logger.warning(f"Invalid scan id format: {scan_id}, error: {e}")
        return jsonify({"error": "Invalid scan id"}), 400

    if not scan:
        current_app.logger.warning(f"Scan not found: {scan_id}")
        return jsonify({"error": "Scan not found"}), 404

    current_app.logger.info(f"Scan found: {scan_id}")
    return jsonify(serialize_scan(scan))


@patient_bp.route("/scans/<scan_id>", methods=["DELETE"])
def delete_scan(scan_id: str):
    """Delete a scan by its ID and optionally remove images from Cloudinary."""
    db = get_db()
    scans_col = db["scans"]
    patients_col = db["patients"]

    scan_id = unquote(scan_id)
    current_app.logger.info(f"Deleting scan with ID: {scan_id}")

    # Find the scan first
    scan = None
    try:
        obj_id = ObjectId(scan_id)
        scan = scans_col.find_one({"_id": obj_id})
    except Exception as e:
        current_app.logger.warning(f"Invalid scan id format: {scan_id}, error: {e}")
        return jsonify({"error": "Invalid scan id"}), 400

    if not scan:
        return jsonify({"error": "Scan not found"}), 404

    patient_id = scan.get("patientId")

    # Try to delete Cloudinary images if URLs are present
    cloudinary_paths = [
        scan.get("imagePath"),
        scan.get("annotatedImagePath"),
        scan.get("gradCamPath"),
    ]
    for img_url in cloudinary_paths:
        if img_url and "cloudinary" in str(img_url).lower():
            try:
                import cloudinary
                import cloudinary.uploader

                cloud_name = current_app.config.get("CLOUDINARY_CLOUD_NAME")
                api_key = current_app.config.get("CLOUDINARY_API_KEY")
                api_secret = current_app.config.get("CLOUDINARY_API_SECRET")

                if all([cloud_name, api_key, api_secret]):
                    cloudinary.config(
                        cloud_name=cloud_name,
                        api_key=api_key,
                        api_secret=api_secret,
                    )
                    # Extract public_id from URL
                    parts = img_url.split("/upload/")
                    if len(parts) > 1:
                        public_id = parts[1].rsplit(".", 1)[0]
                        # Remove version prefix (e.g., v1234567890/)
                        if public_id.startswith("v") and "/" in public_id:
                            public_id = public_id.split("/", 1)[1]
                        cloudinary.uploader.destroy(public_id)
                        current_app.logger.info(f"Deleted Cloudinary image: {public_id}")
            except Exception as cloud_err:
                current_app.logger.warning(f"Failed to delete Cloudinary image: {cloud_err}")

    # Delete the scan document
    scans_col.delete_one({"_id": obj_id})

    # Update patient aggregate (decrement totalScans)
    if patient_id:
        try:
            patient_obj = None
            try:
                p_obj_id = ObjectId(patient_id)
                patient_obj = patients_col.find_one({"_id": p_obj_id})
            except Exception:
                patient_obj = patients_col.find_one({"patientId": patient_id})

            if patient_obj:
                patient_mongo_id = patient_obj.get("_id")
                current_total = patient_obj.get("totalScans", 1)
                new_total = max(0, current_total - 1)

                # Find latest remaining scan for lastScanDate
                latest_scan = scans_col.find_one(
                    {"patientId": patient_id},
                    sort=[("scanDate", -1)],
                )
                new_last_scan_date = latest_scan.get("scanDate") if latest_scan else None

                patients_col.update_one(
                    {"_id": patient_mongo_id},
                    {"$set": {"totalScans": new_total, "lastScanDate": new_last_scan_date}},
                )
        except Exception as e:
            current_app.logger.warning(f"Failed to update patient aggregate after delete: {e}")

    current_app.logger.info(f"Scan deleted successfully: {scan_id}")
    return jsonify({"message": "Scan deleted successfully"}), 200


@patient_bp.route("/patients/<patient_id>", methods=["DELETE"])
def delete_patient(patient_id: str):
    """Delete a patient and all associated scans."""
    db = get_db()
    patients_col = db["patients"]
    scans_col = db["scans"]

    patient_id = unquote(patient_id)
    current_app.logger.info(f"Deleting patient with ID: {patient_id}")

    # Find the patient
    patient = None
    try:
        obj_id = ObjectId(patient_id)
        patient = patients_col.find_one({"_id": obj_id})
    except Exception:
        patient = patients_col.find_one({"patientId": patient_id})

    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    patient_mongo_id = str(patient.get("_id"))

    # Delete all scans for this patient
    scans_col.delete_many({"patientId": patient_mongo_id})

    # Delete the patient
    patients_col.delete_one({"_id": patient.get("_id")})

    current_app.logger.info(f"Patient and scans deleted: {patient_id}")
    return jsonify({"message": "Patient and all scans deleted successfully"}), 200
