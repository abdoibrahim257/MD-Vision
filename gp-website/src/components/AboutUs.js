import React from 'react'
import NavBar from './navbar'
import '../styles/AboutUs.css'


const AboutUs = () => {
  return (
    <div>

        <div className='objective-wrapper'>
            <NavBar />
            <div className='content obj'>
                <h1>Hi, we're MD Vision.</h1>
                <p> Welcome to our platform! We understand how important it is for you to be informed about your health, especially before visiting your doctor. With busy schedules and limited availability, doctors often don't have the time to conduct detailed pre-assessment questionnaires. That's where we come in!</p>
                <p> Our application bridges this gap by providing you with a preliminary understanding of your health conditions using cutting-edge technology. By leveraging computer vision, our tool translates complex medical images into easy-to-understand descriptions. But we don't stop there.</p>
                <p>We also feature a conversational agent that engages with you in a friendly dialogue to gather information about your symptoms. This helps in providing a preliminary differential diagnosis, giving you a clearer picture before your doctor's appointment.</p>
                <p>Our goal is simple: to save doctors' valuable time and empower you to be an informed participant in your healthcare journey. We believe that by keeping you well-informed, we can enhance the overall efficiency of healthcare and boost your satisfaction.</p>
            </div>
        </div>
        <div className='content uss'>
            <div className='picture' id='Abod'>
            </div>
            <div className='picture' id='Amr'>
            </div>
            <div className='picture' id='Amroy'>
            </div>
            <div className='picture' id='emad'>
            </div>

        </div>
        
    </div>
  )
}

export default AboutUs