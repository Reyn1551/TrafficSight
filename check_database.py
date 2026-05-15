import psycopg2
from db_config import DB_CONFIG


def inspect_db():
    print(f"--- MENGAKSES DATABASE PostgreSQL: {DB_CONFIG['dbname']} ---")

    try:
        # 1. Koneksi ke database PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 2. Ambil daftar semua tabel (public schema)
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = cursor.fetchall()

        if not tables:
            print("Database kosong (tidak ada tabel).")

        for table in tables:
            table_name = table[0]
            print(f"\n[TABEL] {table_name}")

            # 3. Ambil info kolom
            cursor.execute(f"""
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_name = %s ORDER BY ordinal_position;
            """, (table_name,))
            columns = cursor.fetchall()
            col_names = [c[0] for c in columns]
            print(f"  Kolom: {', '.join(col_names)}")

            # 4. Ambil sampel data (5 baris pertama)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()

            for row in rows:
                print(f"  Data: {row}")

            # 5. Hitung total row
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total = cursor.fetchone()[0]
            print(f"  Total rows: {total}")

        cursor.close()
        conn.close()
        print("\n--- SELESAI ---")

    except psycopg2.Error as e:
        print(f"[PostgreSQL ERROR] Terjadi kesalahan: {e}")


if __name__ == "__main__":
    inspect_db()