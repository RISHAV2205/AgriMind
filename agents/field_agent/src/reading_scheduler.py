
"""Scheduled generation of simulated field sensor readings."""

from datetime import datetime

from agents.field_agent.src.farm_repository import FarmRepository
from agents.field_agent.src.interface import FieldReading
from agents.field_agent.src.repository import FieldReadingRepository
from agents.field_agent.src.simulator_agent import FieldSimulatorAgent



def generate_readings() -> None:
    """Generate and persist one reading for every registered field."""
    print("Reading scheduler job executed.", flush=True)

    try:
        farm_repository = FarmRepository()
        reading_repository = FieldReadingRepository()
        simulator = FieldSimulatorAgent()

        fields = farm_repository.get_all_fields()
        print(f"Found {len(fields)} fields.", flush=True)

    except Exception as error:
        print(
            f"Failed to initialize reading generation: {error}",
            flush=True,
        )
        return

    for field in fields:
        field_id = str(field["id"])

        try:
            result = simulator.get_field_data(field_id)

            if result.get("status") != "success":
                print(
                    f"Skipped field {field_id}: simulator returned no success",
                    flush=True,
                )
                continue

            data = result["data"]

            reading = FieldReading(
                field_id=data["field_id"],
                soil_moisture_percent=data["soil_moisture_percent"],
                soil_temperature_c=data["soil_temperature_c"],
                soil_ph=data["soil_ph"],
                nitrogen_mg_kg=data["nitrogen_mg_kg"],
                phosphorus_mg_kg=data["phosphorus_mg_kg"],
                potassium_mg_kg=data["potassium_mg_kg"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )

            reading_id = reading_repository.save(reading)

            print(
                f"Saved reading {reading_id} for field {field_id}",
                flush=True,
            )

        except Exception as error:
            print(
                f"Failed to generate reading for field {field_id}: {error}",
                flush=True,
            )
