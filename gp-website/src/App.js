import './App.css'; //here this css is the global css file for the whole website (the temoplate)
import LandingPage from './components/landingPage';
import ChatBot from './components/chatbot';
function App() {
  return (
    <>
      {/* here we will handle all routes AKA the biggest component */}
      <LandingPage />
      {/* <ChatBot /> */}
    </>
  );
}

export default App;
