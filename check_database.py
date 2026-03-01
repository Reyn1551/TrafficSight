import sqlite3
import os

def inspect_db(db_name):
    # Cek apakah file ada
    if not os.path.exists(db_name):
        print(f"[ERROR] File database '{db_name}' tidak ditemukan di folder ini.")
        return

    print(f"--- MENGAKSES DATABASE: {db_name} ---")
    
    try:
        # 1. Koneksi ke database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # 2. Ambil daftar semua tabel
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("Database kosong (tidak ada tabel).")
        
        for table in tables:
            table_name = table[0]
            print(f"\n[TABEL] {table_name}")
            
            # 3. Ambil sampel data (5 baris pertama)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            
            for row in rows:
                print(f"  Data: {row}")

        conn.close()
        print("\n--- SELESAI ---")

    except sqlite3.Error as e:
        print(f"[SQL ERROR] Terjadi kesalahan: {e}")

if __name__ == "__main__":
    # Pastikan nama file sesuai dengan yang kamu sebutkan (sigap_taffic.db)
    inspect_db("sigap_taffic.db")
