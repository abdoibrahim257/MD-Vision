import { Route, Routes } from 'react-router-dom';
import './App.css'; //here this css is the global css file for the whole website (the temoplate)

import LandingPage from './components/landingPage';
import NavBar from './components/navbar';
import SymptomPage from './components/symptomPage';


function App() {
  return (
    <>
      {/* here we will handle all routes AKA the biggest component */}
      <Routes>
        <Route path="/" element={<LandingPage />} />

        <Route path="/maven" element={<SymptomPage />} />

      </Routes>
    </>
  );
}

export default App;
