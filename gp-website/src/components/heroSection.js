import React from 'react';
import '../styles/heroSection.css';
import robot from '../assets/image1.png';
import NavBar from './navbar';


function HeroSection() {
    const [padded, setPadded] = React.useState(false)
  return (
    <div className="background">
        <NavBar setPadding={setPadded}/>
        <div className={padded ? "content pageFront maintain-content": "content pageFront" }>
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