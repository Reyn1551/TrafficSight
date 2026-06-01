# ===============================================================
#  TrafficSight — Konfigurasi Database & Object Storage (MinIO)
# ===============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "trafficsight",
    "user": "trafficsight_user",
    "password": "trafficsight_pass",
}

MINIO_CONFIG = {
    "endpoint": "127.0.0.1:9000",
    "access_key": "admin_storage",
    "secret_key": "password_storage_aman",
    "secure": False,
    "bucket_name": "trafficsight-evidences",
}
