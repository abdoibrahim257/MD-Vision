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
                Experience the future of medical care today. Our innovative AI technology, combined with compassionate healthcare professionals, ensures you receive the most accurate and personalized treatment. Because your health deserves the best of both worlds. 
                </p>
            </div>
            <img src={robot} alt = "Robot representing our project"/>
        </div>
    </div>
  );
}

export default HeroSection;