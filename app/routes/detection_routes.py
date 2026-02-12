from flask import Blueprint, request, jsonify, render_template, current_app
import os
import sys
import tempfile
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
from app.services.yolo_service import detect_image, generate_gradcam
from app.services.gemini_service import describe_detected_image, chatbot_msg
# Paths to scripts (must match sudoers)


detection_bp = Blueprint('detection', __name__)

@detection_bp.route('/')
def index():
    return render_template('index.html')

@detection_bp.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save upload to a temporary directory for processing
    tmp_dir = tempfile.mkdtemp()
    try:
        print(f"[DEBUG] Starting detection for file: {file.filename}")
        filename = secure_filename(file.filename)
        img_path = os.path.join(tmp_dir, filename)
        file.save(img_path)
        print(f"[DEBUG] File saved to: {img_path}")

        # Run detection (model returns result image path, confidence, prediction, model_version, num_detections)
        print("[DEBUG] Running YOLO detection...")
        result_img_path, confidence, prediction, model_version, num_detections = detect_image(img_path, tmp_dir)
        print(f"[DEBUG] Detection completed. Result image: {result_img_path}, Detections: {num_detections}")

        # Generate Grad-CAM visualization
        print("[DEBUG] Generating Grad-CAM visualization...")
        gradcam_img_path = None
        try:
            gradcam_img_path = generate_gradcam(img_path, tmp_dir)
            if gradcam_img_path and os.path.exists(gradcam_img_path):
                print(f"[DEBUG] Grad-CAM generated successfully: {gradcam_img_path}")
            else:
                print("[DEBUG] Grad-CAM generation returned None or file not found")
                gradcam_img_path = None
        except Exception as gradcam_error:
            print(f"[DEBUG] Grad-CAM generation failed: {gradcam_error}")
            import traceback
            traceback.print_exc()
            gradcam_img_path = None

        # Describe image using Gemini
        print("[DEBUG] Generating description with Gemini...")
        try:
            description = describe_detected_image(result_img_path, confidence=confidence, prediction=prediction, num_detections=num_detections)
            print(f"[DEBUG] Description generated: {description[:100]}...")
        except Exception as ge:
            print(f"[DEBUG] Gemini describe failed: {ge}")
            current_app.logger.exception("Gemini describe failed")
            # Use fallback description instead of returning error
            description = "Deskripsi gambar tidak tersedia. Analisis visual menunjukkan deteksi batu ginjal."
        
        analyzed_at = datetime.utcnow().isoformat() + "Z"
        print(f"[DEBUG] Analyzed at: {analyzed_at}", flush=True)
        sys.stderr.write(f"[DEBUG] Analyzed at: {analyzed_at}\n")
        sys.stderr.flush()

        # Import and configure Cloudinary here to avoid module import-time failure
        print("[DEBUG] Importing Cloudinary...", flush=True)
        sys.stderr.write("[DEBUG] Importing Cloudinary...\n")
        sys.stderr.flush()
        cloudinary_available = False
        try:
            import cloudinary
            import cloudinary.uploader
            cloudinary_available = True
            print("[DEBUG] Cloudinary imported successfully")
        except ModuleNotFoundError as me:
            print(f"[DEBUG] Cloudinary module not found: {me}. Using fallback mode (base64).")
            cloudinary_available = False
        except Exception as import_error:
            print(f"[DEBUG] Error importing Cloudinary: {import_error}. Using fallback mode (base64).")
            cloudinary_available = False
        
        # If Cloudinary is not available, return result with base64 images
        if not cloudinary_available:
            print("[DEBUG] Cloudinary not available, using base64 fallback...")
            try:
                import base64
                with open(result_img_path, 'rb') as f:
                    result_bytes = f.read()
                result_b64 = base64.b64encode(result_bytes).decode('ascii')
                ext = os.path.splitext(result_img_path)[1].lower()
                mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                result_data_uri = f"data:{mime};base64,{result_b64}"
                
                with open(img_path, 'rb') as f:
                    orig_bytes = f.read()
                orig_b64 = base64.b64encode(orig_bytes).decode('ascii')
                orig_data_uri = f"data:{mime};base64,{orig_b64}"
            except Exception as b64_error:
                print(f"[DEBUG] Error creating base64: {b64_error}")
                result_data_uri = None
                orig_data_uri = None
            
            # Include Grad-CAM in base64 if available
            gradcam_data_uri = None
            if gradcam_img_path and os.path.exists(gradcam_img_path):
                try:
                    with open(gradcam_img_path, 'rb') as f:
                        gradcam_bytes = f.read()
                    gradcam_b64 = base64.b64encode(gradcam_bytes).decode('ascii')
                    gradcam_ext = os.path.splitext(gradcam_img_path)[1].lower()
                    gradcam_mime = 'image/jpeg' if gradcam_ext in ['.jpg', '.jpeg'] else 'image/png'
                    gradcam_data_uri = f"data:{gradcam_mime};base64,{gradcam_b64}"
                except Exception as gradcam_b64_error:
                    print(f"[DEBUG] Error creating Grad-CAM base64: {gradcam_b64_error}")
                    gradcam_data_uri = None
            
            fallback_response = {
                "result_image": os.path.basename(result_img_path),
                "result_image_data_uri": result_data_uri,
                "original_image": os.path.basename(img_path),
                "original_image_data_uri": orig_data_uri,
                "description": description,
                "confidence": confidence,
                "prediction": prediction,
                "model_version": model_version,
                "analyzed_at": analyzed_at,
                "cloudinary_failed": True,
                "cloudinary_error": "Cloudinary package not installed. Install with 'pip install cloudinary'.",
                "original_image_path": None,
                "annotated_image_path": None,
                "gradcam_image": os.path.basename(gradcam_img_path) if gradcam_img_path else None,
                "gradcam_path": None,
                "gradcam_image_data_uri": gradcam_data_uri,
            }
            print("[DEBUG] Returning fallback response (base64)")
            return jsonify(fallback_response)

        # Configure Cloudinary using app config
        print("[DEBUG] Configuring Cloudinary...")
        try:
            cloudinary_cloud_name = current_app.config.get("CLOUDINARY_CLOUD_NAME")
            cloudinary_api_key = current_app.config.get("CLOUDINARY_API_KEY")
            cloudinary_api_secret = current_app.config.get("CLOUDINARY_API_SECRET")
            print(f"[DEBUG] Cloudinary config retrieved - cloud_name: {cloudinary_cloud_name is not None}, api_key: {cloudinary_api_key is not None}, api_secret: {cloudinary_api_secret is not None}")
        except Exception as config_error:
            print(f"[DEBUG] Error getting Cloudinary config: {config_error}")
            return (
                jsonify({
                    "error": f"Error membaca konfigurasi Cloudinary: {str(config_error)}"
                }),
                500,
            )
        
        if not all([cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret]):
            print("[DEBUG] Cloudinary credentials missing!")
            current_app.logger.error("Cloudinary credentials missing in config")
            return (
                jsonify({
                    "error": "Cloudinary configuration tidak lengkap. Pastikan CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, dan CLOUDINARY_API_SECRET sudah diatur."
                }),
                500,
            )
        
        try:
            cloudinary.config(
                cloud_name=cloudinary_cloud_name,
                api_key=cloudinary_api_key,
                api_secret=cloudinary_api_secret,
            )
            print("[DEBUG] Cloudinary configured successfully")
        except Exception as config_set_error:
            print(f"[DEBUG] Error setting Cloudinary config: {config_set_error}")
            return (
                jsonify({
                    "error": f"Error mengatur konfigurasi Cloudinary: {str(config_set_error)}"
                }),
                500,
            )

        # Upload original and annotated images to Cloudinary
        print("[DEBUG] Starting Cloudinary uploads...")
        try:
            # Verify files exist before uploading
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Original image not found: {img_path}")
            if not os.path.exists(result_img_path):
                raise FileNotFoundError(f"Result image not found: {result_img_path}")
            
            print(f"[DEBUG] Uploading original image: {img_path}")
            current_app.logger.info(f"Uploading original image to Cloudinary: {img_path}")
            orig_upload = cloudinary.uploader.upload(
                img_path, 
                folder="kidneystone/originals", 
                use_filename=True, 
                unique_filename=False
            )
            print(f"[DEBUG] Original uploaded: {orig_upload.get('secure_url')}")
            current_app.logger.info(f"Original image uploaded successfully: {orig_upload.get('secure_url')}")
            
            print(f"[DEBUG] Uploading annotated image: {result_img_path}")
            current_app.logger.info(f"Uploading annotated image to Cloudinary: {result_img_path}")
            annotated_upload = cloudinary.uploader.upload(
                result_img_path, 
                folder="kidneystone/annotated", 
                use_filename=True, 
                unique_filename=True
            )
            print(f"[DEBUG] Annotated uploaded: {annotated_upload.get('secure_url')}")
            current_app.logger.info(f"Annotated image uploaded successfully: {annotated_upload.get('secure_url')}")
            
            # Upload Grad-CAM image if available
            gradcam_upload = None
            if gradcam_img_path and os.path.exists(gradcam_img_path):
                print(f"[DEBUG] Uploading Grad-CAM image: {gradcam_img_path}")
                current_app.logger.info(f"Uploading Grad-CAM image to Cloudinary: {gradcam_img_path}")
                try:
                    gradcam_upload = cloudinary.uploader.upload(
                        gradcam_img_path,
                        folder="kidneystone/gradcam",
                        use_filename=True,
                        unique_filename=True
                    )
                    print(f"[DEBUG] Grad-CAM uploaded: {gradcam_upload.get('secure_url')}")
                    current_app.logger.info(f"Grad-CAM image uploaded successfully: {gradcam_upload.get('secure_url')}")
                except Exception as gradcam_upload_error:
                    print(f"[DEBUG] Grad-CAM upload failed: {gradcam_upload_error}")
                    current_app.logger.exception("Grad-CAM upload failed")
                    gradcam_upload = None
            else:
                print("[DEBUG] Grad-CAM image not available for upload")
        except Exception as ce:
            print(f"[DEBUG] Cloudinary upload failed: {ce}")
            # If Cloudinary upload fails, log and return a fallback response
            current_app.logger.exception("Cloudinary upload failed")
            try:
                import base64

                with open(result_img_path, 'rb') as f:
                    b = f.read()
                b64 = base64.b64encode(b).decode('ascii')
                ext = os.path.splitext(result_img_path)[1].lower()
                mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                data_uri = f"data:{mime};base64,{b64}"
            except Exception:
                data_uri = None

            # Include Grad-CAM in base64 if available
            gradcam_data_uri = None
            if gradcam_img_path and os.path.exists(gradcam_img_path):
                try:
                    with open(gradcam_img_path, 'rb') as f:
                        gradcam_bytes = f.read()
                    gradcam_b64 = base64.b64encode(gradcam_bytes).decode('ascii')
                    gradcam_ext = os.path.splitext(gradcam_img_path)[1].lower()
                    gradcam_mime = 'image/jpeg' if gradcam_ext in ['.jpg', '.jpeg'] else 'image/png'
                    gradcam_data_uri = f"data:{gradcam_mime};base64,{gradcam_b64}"
                except Exception as gradcam_b64_error:
                    print(f"[DEBUG] Error creating Grad-CAM base64: {gradcam_b64_error}")
                    gradcam_data_uri = None
            
            # Return YOLO result + description but indicate Cloudinary failed
            fallback_response = {
                "result_image": os.path.basename(result_img_path),
                "result_image_data_uri": data_uri,
                "original_image": os.path.basename(img_path),
                "description": description,
                "confidence": confidence,
                "prediction": prediction,
                "model_version": model_version,
                "analyzed_at": analyzed_at,
                "cloudinary_failed": True,
                "cloudinary_error": str(ce),
                "gradcam_image": os.path.basename(gradcam_img_path) if gradcam_img_path else None,
                "gradcam_path": None,
                "gradcam_image_data_uri": gradcam_data_uri,
            }
            return jsonify(fallback_response)

        # Validate upload results
        print("[DEBUG] Validating upload results...")
        if not orig_upload or "secure_url" not in orig_upload:
            print(f"[DEBUG] Original upload invalid: {orig_upload}")
            current_app.logger.error("Original image upload did not return secure_url")
            raise ValueError("Gagal mendapatkan URL gambar original dari Cloudinary")
        
        if not annotated_upload or "secure_url" not in annotated_upload:
            print(f"[DEBUG] Annotated upload invalid: {annotated_upload}")
            current_app.logger.error("Annotated image upload did not return secure_url")
            raise ValueError("Gagal mendapatkan URL gambar hasil analisis dari Cloudinary")
        
        print("[DEBUG] Building response data...")
        # Get Grad-CAM URL if available
        gradcam_url = None
        gradcam_image_name = None
        gradcam_data_uri = None
        
        if gradcam_upload and "secure_url" in gradcam_upload:
            gradcam_url = gradcam_upload.get("secure_url")
            gradcam_image_name = os.path.basename(gradcam_img_path) if gradcam_img_path else None
            print(f"[DEBUG] Grad-CAM URL from Cloudinary: {gradcam_url}")
        elif gradcam_img_path and os.path.exists(gradcam_img_path):
            # If Cloudinary upload failed but file exists, create base64 fallback
            gradcam_image_name = os.path.basename(gradcam_img_path)
            print(f"[DEBUG] Grad-CAM file exists but Cloudinary upload failed, creating base64 fallback")
            try:
                import base64
                with open(gradcam_img_path, 'rb') as f:
                    gradcam_bytes = f.read()
                gradcam_b64 = base64.b64encode(gradcam_bytes).decode('ascii')
                gradcam_ext = os.path.splitext(gradcam_img_path)[1].lower()
                gradcam_mime = 'image/jpeg' if gradcam_ext in ['.jpg', '.jpeg'] else 'image/png'
                gradcam_data_uri = f"data:{gradcam_mime};base64,{gradcam_b64}"
                print(f"[DEBUG] Grad-CAM base64 created successfully")
            except Exception as gradcam_b64_error:
                print(f"[DEBUG] Error creating Grad-CAM base64: {gradcam_b64_error}")
                gradcam_data_uri = None
        
        print(f"[DEBUG] Final Grad-CAM data - URL: {gradcam_url}, Name: {gradcam_image_name}, DataURI: {'present' if gradcam_data_uri else 'none'}")
        
        response_data = {
            "result_image": os.path.basename(result_img_path),
            "result_image_url": annotated_upload.get("secure_url"),
            "original_image": os.path.basename(img_path),
            "original_image_url": orig_upload.get("secure_url"),
            "description": description,
            "confidence": confidence,
            "prediction": prediction,
            "model_version": model_version,
            "analyzed_at": analyzed_at,
            # Provide path fields as full URLs (frontend will accept absolute URLs)
            "original_image_path": orig_upload.get("secure_url"),
            "annotated_image_path": annotated_upload.get("secure_url"),
            "gradcam_image": gradcam_image_name,
            "gradcam_path": gradcam_url,
            "gradcam_image_data_uri": gradcam_data_uri,
        }

        print("[DEBUG] Detection completed successfully!")
        current_app.logger.info("Detection completed successfully")
        return jsonify(response_data)
    except Exception as e:
        # Log full traceback to server logs for debugging
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[DEBUG] ERROR in /detect route:", flush=True)
        print(error_traceback, flush=True)
        sys.stderr.write(f"[DEBUG] ERROR in /detect route:\n")
        sys.stderr.write(error_traceback)
        sys.stderr.flush()
        try:
            current_app.logger.exception(f"Error in /detect route: {str(e)}")
        except Exception:
            traceback.print_exc()

        # Return more detailed error information in response to aid debugging
        error_message = str(e)
        error_type = type(e).__name__
        
        # Handle specific error types dengan pesan yang lebih user-friendly
        if isinstance(e, FileNotFoundError) or "FileNotFoundError" in error_type:
            # Pesan error yang jelas dan actionable untuk frontend
            if "Model YOLO" in error_message or "best.pt" in error_message:
                error_message = "Model YOLO tidak ditemukan. Silakan hubungi administrator untuk menambahkan file model (best.pt) ke server."
            else:
                error_message = "File yang diperlukan tidak ditemukan. Silakan hubungi administrator."
        elif "Cloudinary" in error_message or "cloudinary" in error_message.lower():
            error_message = f"Error saat mengunggah ke Cloudinary: {error_message}"
        elif "Gemini" in error_message or "gemini" in error_message.lower():
            error_message = f"Error saat analisis dengan Gemini: {error_message}"
        elif "YOLO" in error_message or "detection" in error_message.lower() or "detect_image" in error_message:
            error_message = f"Error saat deteksi gambar: {error_message}"
        
        return (
            jsonify({
                "error": error_message,
                "error_type": type(e).__name__,
                "traceback": error_traceback if current_app.config.get("DEBUG") else None,
            }),
            500,
        )
    finally:
        # Clean temporary files
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


@detection_bp.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        if(data is None):
            return jsonify({"error": "No image uploaded"}), 400
        msg = data.get('message')
        descript = data.get('descript')

        res_msg = chatbot_msg(msg, descript)

        return jsonify({
            "result_msg": res_msg
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@detection_bp.route('/detect-debug', methods=['POST'])
def detect_debug():
    """Temporary debug endpoint: only runs YOLO detection and returns metadata
    and the annotated image as a base64 data URI. Use this to isolate whether
    detection is working before Cloudinary/Gemini steps.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        filename = secure_filename(file.filename)
        img_path = os.path.join(tmp_dir, filename)
        file.save(img_path)

        # Run detection
        result_img_path, confidence, prediction, model_version, num_detections = detect_image(img_path, tmp_dir)

        # Read annotated image and encode as data URI
        try:
            import base64

            with open(result_img_path, 'rb') as f:
                b = f.read()
            b64 = base64.b64encode(b).decode('ascii')
            # Try to infer mime type from extension
            ext = os.path.splitext(result_img_path)[1].lower()
            mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
            data_uri = f"data:{mime};base64,{b64}"
        except Exception as e:
            data_uri = None

        return jsonify({
            "result_image": os.path.basename(result_img_path),
            "result_image_data_uri": data_uri,
            "confidence": confidence,
            "prediction": prediction,
            "model_version": model_version,
        })
    except Exception as e:
        current_app.logger.exception("Error in /detect-debug route")
        return jsonify({"error": str(e), "error_type": type(e).__name__}), 500
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


@detection_bp.route('/model-status', methods=['GET'])
def model_status():
    """Endpoint untuk mengecek status model YOLO"""
    try:
        import os
        model_path = 'app/model/best.pt'
        abs_path = os.path.abspath(model_path)
        exists = os.path.exists(model_path)
        
        status = {
            "model_available": exists,
            "model_path": abs_path,
            "message": "Model YOLO siap digunakan" if exists else "Model YOLO tidak ditemukan"
        }
        
        if not exists:
            status["instructions"] = "Silakan tambahkan file best.pt ke folder backend/app/model/"
        
        return jsonify(status), 200 if exists else 404
    except Exception as e:
        return jsonify({
            "model_available": False,
            "error": str(e),
            "message": "Error saat mengecek status model"
        }), 500


@detection_bp.route('/mongo-test', methods=['GET'])
def mongo_test():
    try:
        db = current_app.config.get('MONGO_DB')
        if db is None:
            return jsonify({"error": "MongoDB is not configured"}), 500

        result = db['test_collection'].insert_one({
            "message": "MongoDB connection OK",
            "created_at": datetime.utcnow()
        })

        return jsonify({
            "inserted_id": str(result.inserted_id)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500