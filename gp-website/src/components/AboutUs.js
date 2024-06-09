import React, { useState } from 'react'
import NavBar from './navbar'
import '../styles/AboutUs.css'
import linkedin from '../assets/linkedin.svg'
import github from '../assets/github.svg'
import email from '../assets/email.svg'

const AboutUs = () => {
  const [padded, setPadding] = useState(false)
  return (
    <div>
        <div className='objective-wrapper'>
            <NavBar setPadding={setPadding}/>
            <h1 className={padded ? 'content title maintain-content' : 'content title'} id='page-top'>Hi, we're <span id = "md">MD</span><spav id = "vision">Vision</spav>.</h1>
            <hr id='margin'/>
            <div className='content obj'>
                <p> Welcome to our platform! We understand how important it is for you to be informed about your health, especially before visiting your doctor. With busy schedules and limited availability, doctors often don't have the time to conduct detailed pre-assessment questionnaires. That's where we come in!</p>
                <p> Our application bridges this gap by providing you with a preliminary understanding of your health conditions using cutting-edge technology. By leveraging computer vision, our tool translates complex medical images into easy-to-understand descriptions. But we don't stop there.</p>
                <p>We also feature a conversational agent that engages with you in a friendly dialogue to gather information about your symptoms. This helps in providing a preliminary differential diagnosis, giving you a clearer picture before your doctor's appointment.</p>
                <p>Our goal is simple: to save doctors' valuable time and empower you to be an informed participant in your healthcare journey. We believe that by keeping you well-informed, we can enhance the overall efficiency of healthcare and boost your satisfaction.</p>
            </div>
        </div>
        <div className='content us-wrapper'>
            <h1 className='title'>Meet the team</h1>
            <hr id='margin'/>
            <div class="uss">
              <div class="individual">
                <div className='picture' id='Abod'></div>
                <div>
                  <p id='name'>Abdulrahman Ibrahim</p>
                  <div className='contact'>
                    <a href='https://www.linkedin.com/in/abdulrahman-elhosseini-29657321a/'><img className="interact" id='linkedin' src={linkedin} alt='linkedIn link to Abdulrahman'/></a>
                    <a href='https://github.com/abdoibrahim257?tab=overview&from=2024-06-01&to=2024-06-07'><img className="interact" id='github' src={github} alt='github link to Abdulrahman'/></a>
                    <a href='mailto:abdoibrahim257@gmail.com?subject=Reach From MDVision'><img className="interact" id='email' src={email} alt='email link to Abdulrahman'/></a>
                  </div>
                </div>
              </div>
              <div class="individual">
                <div className='picture' id='emad'></div>
                <div>
                  <p id='name'>Ahmed Emad</p>
                  <div className='contact'>
                    <a href='https://www.linkedin.com/in/ahmed-emad-02a306221/'><img className="interact" id='linkedin' src={linkedin} alt='linkedIn link to Ahmed Emad'/></a>
                    <a href='https://github.com/ahmedemad81'><img className="interact" id='github' src={github} alt='github link to Ahmed Emad'/></a>
                    <a href='mailto:ahmedemad8@gmail.com?subject=Reach From MDVision'><img className="interact" id='email' src={email} alt='email link to Ahmed Emad'/></a>
                  </div>
                </div>
              </div>
              <div class="individual">
                <div className='picture' id='Amr'></div>
                <div>
                  <p id='name'>Amr Ahmed</p>
                  <div className='contact'>
                    <a href='https://www.linkedin.com/in/amr--ahmed/'><img className="interact" id='linkedin' src={linkedin} alt='linkedIn link to Amr Ahmed'/></a>
                    <a href='https://github.com/AmrAhmed412'><img className="interact" id='github' src={github} alt='github link to Amr Ahmed'/></a>
                    <a href='mailto:amr23034@hotmail.com?subject=Reach From MDVision'><img className="interact" id='email' src={email} alt='email link to Amr Ahmed'/></a>
                  </div>
                </div>
              </div>
              <div class="individual">
                <div className='picture' id='Amroy'></div>
                <div>
                  <p id='name'>Amr Yasser</p>
                  <div className='contact'>
                    <a href='https://www.linkedin.com/in/amryhassan/'><img className="interact" id='linkedin' src={linkedin} alt='linkedIn link to Amr Yasser'/></a>
                    <a href='https://github.com/AmroY1209'><img className="interact" id='github' src={github} alt='github link to Amr Yasser'/></a>
                    <a href='mailto:amryasser1209@gmail.com?subject=Reach From MDVision'><img className="interact" id='email' src={email} alt='email link to Amr Yasser'/></a>
                  </div>
                </div>
              </div>
            </div>
            <p className='quote'>
              "We are a dedicated team committed to innovation and excellence. 
              Each of us brings unique talents and perspectives, making our collaboration both 
              dynamic and effective. Together, we strive to deliver exceptional results and 
              drive our mission forward. Meet the people who make it all happen."
            </p>
        </div>
    </div>
  )
}

export default AboutUs