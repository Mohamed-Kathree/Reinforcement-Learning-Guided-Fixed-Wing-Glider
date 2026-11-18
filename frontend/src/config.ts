// config.ts
// =========
// Where the FastAPI backend lives. Set VITE_API_BASE (e.g. in frontend/.env.local)
// to run the dashboard against a backend on another machine:
//
//     VITE_API_BASE=http://192.168.1.20:8000
//
// Default: the backend on this machine.
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");
export const WS_BASE: string = API_BASE.replace(/^http/, "ws");
