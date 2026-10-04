from agents.field_agent.data.simulator import FieldDataSimulator
from agents.field_agent.src.repository import FieldReadingRepository


FIELD_ID = "1"

simulator = FieldDataSimulator()
repository = FieldReadingRepository()

# Get the latest reading stored in PostgreSQL.
previous = repository.get_latest(FIELD_ID)

# Generate the next simulated reading.
reading = simulator.get_reading(
    FIELD_ID,
    previous,
)

# Store the new reading.
reading_id = repository.save(reading)

print("Reading saved successfully!")
print(f"Database ID: {reading_id}")
print(f"Field ID: {reading.field_id}")
print(f"Moisture: {reading.soil_moisture_percent}%")
print(f"Temperature: {reading.soil_temperature_c}°C")
print(f"pH: {reading.soil_ph}")
print(f"Nitrogen: {reading.nitrogen_mg_kg}")
print(f"Phosphorus: {reading.phosphorus_mg_kg}")
print(f"Potassium: {reading.potassium_mg_kg}")