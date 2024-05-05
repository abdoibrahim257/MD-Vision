import './App.css'; //here this css is the global css file for the whole website (the temoplate)
import LandingPage from './components/landingPage';
import SymptomPage from './components/symptomPage';
function App() {
  return (
    <>
      {/* here we will handle all routes AKA the biggest component */}
      {/* <LandingPage /> */}
      <SymptomPage />
    </>
  );
}

export default App;
