# TrafficSight

## Migrasi ke PostgreSQL

1. Pastikan PostgreSQL berjalan dan database `trafficsight` tersedia.
2. Perbarui `DATABASE_URL` di `config.py` dengan kredensial yang sesuai.
   Contoh:
   ```python
   DATABASE_URL = "postgresql://username:password@localhost:5432/trafficsight"
   ```
3. Atau set environment variable sebelum migrasi:
   ```bash
   export DATABASE_URL='postgresql://username:password@localhost:5432/trafficsight'
   ```
4. Jalankan migrasi:

```bash
python3 migrate_sqlite_to_postgres.py
```

4. Setelah migrasi selesai, aplikasi akan menyimpan data ke PostgreSQL.
