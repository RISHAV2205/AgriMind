# PostgreSQL setup

Run the bootstrap script to create the local `agrimind` database, apply the
schema, and seed the initial Farmer/Field data:

```powershell
.\venv\Scripts\python.exe database\bootstrap.py
```

Before running it, copy `.env.example` to `.env` and set the actual PostgreSQL
password. The application will
   use `database.connection.create_database_engine()` when persistence is added.

The current schema contains only `farmers` and `fields`. Field readings,
predictions, risks, and alerts will be added as later migrations.
