# Kitchennaire backend

This is a minimal development backend for Kitchennaire. It exposes a simple endpoint that accepts a JSON body with a `yt_url` field and prints it to the console.

Quick start (Windows PowerShell):

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Run the server (development)

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. Test with PowerShell

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/submit_url -Body (ConvertTo-Json @{yt_url = 'https://www.youtube.com/watch?v=XXXX'}) -ContentType 'application/json'
```

Or with curl:

```powershell
curl -X POST http://localhost:8000/submit_url -H "Content-Type: application/json" -d '{"yt_url":"https://www.youtube.com/watch?v=XXXX"}'
```

Notes:
- The server prints the received URL to stdout and logs it. For mobile app testing (Expo), replace "localhost" with your machine IP (such as `192.168.x.y`) so the device/emulator can reach it.
- CORS is permissive for convenience; restrict origins before deploying.
