import React, { useEffect } from 'react';
import NavBar from './navbar'
import '../styles/ScanPage.css'

const ScanPage = () => {

  return (
    <div>
        <div class="scan-background">
            <NavBar/>
            <div class="content">
                <div className='hero-wrapper'>
                    <div className='page-quote'>
                        <p><span id='upload'>Upload</span> a scan</p>
                        <p><span id='ease'>Ease</span> your mind</p>
                        <p>Let's keep your health journey <span id='free'>worry-free!</span></p>
                    </div>
                    <div className='lottie'>
                        <lottie-player
                            // width="30%"
                            autoplay
                            // loop
                            mode="normal"
                            speed = "1.5"
                            // src="https://lottie.host/d73b52cc-4991-44cf-b125-45e96577b4bc/or62hvWNOo.json"
                            src="https://lottie.host/b911b2fd-06fd-4d06-86f7-beb19feb7a6b/oLUWaaFmk1.json"
                        ></lottie-player>
                    </div>
                </div>
            </div>
        </div>
    </div>
  )
}

export default ScanPage