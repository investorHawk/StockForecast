setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r backend/requirements.txt
	cd frontend && npm install

dev:
	. .venv/bin/activate && uvicorn backend.app:app --host 0.0.0.0 --port $${API_PORT:-8082} --reload & \
	cd frontend && npm run dev -- --port $${WEB_PORT:-4002}

test:
	. .venv/bin/activate && pytest -q

