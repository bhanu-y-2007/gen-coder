import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import SessionConfiguration from "./pages/SessionConfiguration";
import SupportConsole from "./pages/SupportConsole";
import SessionResult from "./pages/SessionResult";
import Analytics from "./pages/Analytics";

function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* Opening the project lands directly on the Task 6 Customer
            Configuration screen (persona / scenario / initial emotion /
            severity / patience). No manual URL or session ID typing. */}
        <Route path="/" element={<Navigate to="/session/new" replace />} />

        <Route path="/dashboard" element={<Dashboard />} />

        <Route
          path="/session/new"
          element={<SessionConfiguration />}
        />

        <Route
          path="/session/:sessionId"
          element={<SupportConsole />}
        />

        <Route
          path="/session/:sessionId/result"
          element={<SessionResult />}
        />

        <Route
          path="/analytics"
          element={<Analytics />}
        />

      </Routes>
    </BrowserRouter>
  );
}

export default App;