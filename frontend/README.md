# AgriMind dashboard starter

Minimal vanilla HTML/CSS/JavaScript frontend. No Node.js or build step required.

## Files
- `index.html` — dashboard structure
- `styles.css` — responsive layout and styles
- `app.js` — FastAPI requests, polling, image upload and result rendering

## API endpoints expected
- `GET /api/v1/farmers`
- `GET /api/v1/farmers/{farmer_id}/fields`
- `GET /api/v1/fields/{field_id}?location=...`
- `POST /api/v1/fields/{field_id}/analyze` with multipart fields `image`, `crop`, and optional `location`

## Windows setup
1. Extract these files into `D:\AgriMind\frontend`.
2. Start PostgreSQL and your existing FastAPI app from `D:\AgriMind`. If your app is `api.main:app`, use:
   `python -m uvicorn api.main:app --reload`
   Use the actual existing entry point if it differs.
3. Verify endpoints in `http://127.0.0.1:8000/docs`.
4. In a second PowerShell terminal:
   `cd D:\AgriMind\frontend`
   `python -m http.server 5500`
5. Open `http://127.0.0.1:5500`.

## CORS note
Port 5500 and port 8000 are different origins. If the browser console shows CORS errors, configure FastAPI `CORSMiddleware` for local development with origins `http://127.0.0.1:5500` and `http://localhost:5500`. Do not enable unrestricted origins with credentials in production.

## Notes
- The selected field refreshes every 15 seconds. This is polling, not true streaming.
- The UI shows treatment/advice only when the backend returns it; it does not fabricate recommendations.
- This is a development dashboard and does not implement authentication.
