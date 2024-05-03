import NavBar from "./navbar";
import '../styles/heroSection.css';

import robot from '../assets/image1.png';


function HeroSection() {
  return (
    <div className="background">
        <NavBar />
        <div className="content pageFront">
            <div className="catchphrase">
                <p className="slogan">
                    Increasing Access.<br/>
                    Lowering Costs.<br/>
                    Improving Health.
                </p>
                <p className="goal">
                    Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
                </p>
            </div>
            <img src={robot} alt = "Robot representing our project"/>
        </div>
    </div>
  );
}

export default HeroSection;