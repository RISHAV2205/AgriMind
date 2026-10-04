-- AgriMind's initial PostgreSQL entities.
-- A farmer can own many fields; each field belongs to exactly one farmer.

CREATE TABLE IF NOT EXISTS farmers (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fields (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    farmer_id BIGINT NOT NULL REFERENCES farmers(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (farmer_id, name)
);

-- Initial development data only.
INSERT INTO farmers (name)
VALUES
    ('Farmer A'),
    ('Farmer B'),
    ('Farmer C')
ON CONFLICT (name) DO NOTHING;

INSERT INTO fields (farmer_id, name)
VALUES
    ((SELECT id FROM farmers WHERE name = 'Farmer A'), 'Field 1'),
    ((SELECT id FROM farmers WHERE name = 'Farmer A'), 'Field 2'),
    ((SELECT id FROM farmers WHERE name = 'Farmer B'), 'Field 3'),
    ((SELECT id FROM farmers WHERE name = 'Farmer B'), 'Field 4'),
    ((SELECT id FROM farmers WHERE name = 'Farmer C'), 'Field 5')
ON CONFLICT (farmer_id, name) DO NOTHING;
