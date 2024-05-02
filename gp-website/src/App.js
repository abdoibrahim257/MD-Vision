import './App.css'; //here this css is the global css file for the whole website (the temoplate)
import NavBar from './components/navbar.js'

function App() {
  return (
    <>
      {/* here we will handle all routes AKA the biggest component */}
      {/* <h1>HELLO WORLD</h1>
      <p>Hello this is a test file</p> */}
      <NavBar />
    </>
  );
}

export default App;
