import { Route, Routes } from 'react-router-dom';
import './App.css'; //here this css is the global css file for the whole website (the temoplate)

import LandingPage from './components/landingPage';
import SymptomPage from './components/symptomPage';
import ChatPage from './components/ChatPage';
import ScanPage from './components/ScanPage';
import AboutUs from './components/AboutUs';


function App() {
  return (
    <>
      {/* here we will handle all routes AKA the biggest component */}
      <Routes>
        <Route path="/" element={<LandingPage />} />

        <Route path="/maven" element={<SymptomPage />} />

        <Route path="/maven/:symptom" element={<ChatPage />} />
        {/* <ChatPage /> */}

        <Route path="/upload" element={<ScanPage/>}/>

        <Route path="/about" element={<AboutUs />} />
      </Routes>
    </>
  );
}

export default App;
