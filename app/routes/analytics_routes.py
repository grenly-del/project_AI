from flask import Blueprint, jsonify, current_app
from datetime import datetime, timedelta
from bson import ObjectId
from app.routes.patient_routes import get_db
import pytz

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route("/analytics/classifications", methods=["GET"])
def get_classifications_analytics():
    """
    Get analytics data for classifications including:
    - Total scans
    - Healthy vs CKD counts
    - Distribution
    - Accuracy over time (last 6 months)
    """
    try:
        db = get_db()
        scans_col = db["scans"]
        
        # Get all scans
        all_scans = list(scans_col.find({}))
        
        total_scans = len(all_scans)
        
        # Count healthy vs CKD
        # Assuming prediction field contains the classification
        healthy_count = 0
        ckd_count = 0
        
        for scan in all_scans:
            prediction = scan.get("prediction", "").lower()
            # Adjust these conditions based on your actual prediction values
            if "healthy" in prediction or "normal" in prediction or "no" in prediction:
                healthy_count += 1
            else:
                ckd_count += 1
        
        # Distribution data
        distribution = [
            {
                "label": "Healthy",
                "count": healthy_count,
                "color": "#22c55e"
            },
            {
                "label": "CKD",
                "count": ckd_count,
                "color": "#f87171"
            }
        ]
        
        # Accuracy over time (last 6 months)
        wita = pytz.timezone('Asia/Makassar')
        now = datetime.now(wita)
        
        # Generate last 6 months data
        accuracy_over_time = []
        for i in range(5, -1, -1):  # Last 6 months (5 months ago to current month)
            # Calculate month start
            target_month = now.month - i
            target_year = now.year
            
            # Handle year rollover
            while target_month <= 0:
                target_month += 12
                target_year -= 1
            while target_month > 12:
                target_month -= 12
                target_year += 1
            
            month_start = wita.localize(datetime(target_year, target_month, 1, 0, 0, 0))
            
            # Calculate month end
            if target_month == 12:
                month_end = wita.localize(datetime(target_year + 1, 1, 1, 0, 0, 0)) - timedelta(seconds=1)
            else:
                month_end = wita.localize(datetime(target_year, target_month + 1, 1, 0, 0, 0)) - timedelta(seconds=1)
            
            # For current month, use now as end
            if i == 0:
                month_end = now
            
            # Filter scans by month
            month_scans_filtered = []
            for scan in all_scans:
                scan_date = scan.get("scanDate")
                if not scan_date:
                    continue
                
                # Parse scan date
                if isinstance(scan_date, str):
                    try:
                        # Try ISO format
                        scan_date = datetime.fromisoformat(scan_date.replace('Z', '+00:00'))
                    except:
                        try:
                            # Try other formats
                            scan_date = datetime.strptime(scan_date, "%Y-%m-%d %H:%M:%S")
                            scan_date = wita.localize(scan_date)
                        except:
                            continue
                elif isinstance(scan_date, datetime):
                    # Make timezone aware if not already
                    if scan_date.tzinfo is None:
                        scan_date = wita.localize(scan_date)
                    else:
                        scan_date = scan_date.astimezone(wita)
                else:
                    continue
                
                # Check if scan is in this month
                if month_start <= scan_date <= month_end:
                    month_scans_filtered.append(scan)
            
            # Calculate average confidence for this month
            if month_scans_filtered:
                confidences = []
                for s in month_scans_filtered:
                    conf = s.get("confidence")
                    if conf is not None:
                        try:
                            conf_float = float(conf)
                            # Convert to percentage if needed (if confidence is 0-1, multiply by 100)
                            if conf_float <= 1:
                                conf_float = conf_float * 100
                            confidences.append(conf_float)
                        except (ValueError, TypeError):
                            continue
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            else:
                avg_confidence = 0
            
            # Format date for display (use month name)
            date_str = month_start.strftime("%b %Y")
            
            accuracy_over_time.append({
                "date": date_str,
                "averageConfidence": round(avg_confidence, 2)
            })
        
        return jsonify({
            "totalScans": total_scans,
            "healthyCount": healthy_count,
            "ckdCount": ckd_count,
            "distribution": distribution,
            "accuracyOverTime": accuracy_over_time
        })
        
    except Exception as e:
        current_app.logger.exception(f"Error in analytics: {e}")
        return jsonify({"error": f"Failed to get analytics: {str(e)}"}), 500

